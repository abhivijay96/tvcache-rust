use std::{fs::File, io::BufReader, path::Path};
use tvcache::{
    candidates::CandidateDump,
    policy::{
        classic::ClassicPolicy,
        features::{FeatureSet, BASIC_NAMES, RICH_NAMES},
        linear::{LinearPolicy, LinearWeights},
        mlp::{Activation, Layer, MlpModel, MlpPolicy, Output, FORMAT},
        synthetic::*,
    },
    replay::{check_decisions, replay, replay_observed},
};

fn trace(name: &str) -> (TraceCatalog, Vec<AccessRecord>) {
    let dir = Path::new("data/synthetic").join(name);
    let catalog = TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl")).unwrap()))
        .unwrap();
    let events = read_jsonl(BufReader::new(File::open(dir.join("events.jsonl")).unwrap())).unwrap();
    (catalog, events)
}

fn weights() -> LinearWeights {
    LinearWeights {
        frequency: 0.7,
        recency: 1.3,
        runtime: -0.4,
        dollar_cost: 0.2,
        inverse_size: 0.9,
        depth: -1.1,
        branching: 0.5,
        subtree_frequency: 0.3,
        subtree_recency: 1.7,
    }
}

/// One linear layer over the basic features: an MLP spelling of `LinearPolicy`.
fn linear_as_mlp(w: &LinearWeights) -> MlpModel {
    MlpModel {
        format: FORMAT.into(),
        name: "mlp-linear".into(),
        features: FeatureSet::Basic,
        feature_names: BASIC_NAMES.iter().map(|s| s.to_string()).collect(),
        input_mean: vec![0.0; 9],
        input_std: vec![1.0; 9],
        layers: vec![Layer {
            weight: vec![vec![
                w.frequency,
                w.recency,
                w.runtime,
                w.dollar_cost,
                w.inverse_size,
                w.depth,
                w.branching,
                w.subtree_frequency,
                w.subtree_recency,
            ]],
            bias: vec![0.0],
            activation: Activation::None,
        }],
        output: Output::Score,
        protect_incoming: false,
        max_candidates: None,
        metadata: serde_json::Value::Null,
    }
}

#[tokio::test]
async fn single_layer_mlp_reproduces_linear_policy_decisions() {
    let (catalog, events) = trace("balanced");
    let linear = replay(&catalog, &events, 1, Some(Box::new(LinearPolicy::new(weights()).unwrap())))
        .await
        .unwrap();
    let mlp = MlpPolicy::new(linear_as_mlp(&weights())).unwrap();
    let learned = replay(&catalog, &events, 1, Some(Box::new(mlp))).await.unwrap();
    check_decisions(&learned, &catalog, &events).unwrap();
    assert!(linear.evicted_entries > 0);
    assert_eq!(linear.hits, learned.hits);
    assert_eq!(linear.evicted_entries, learned.evicted_entries);
    for (a, b) in linear.decisions.iter().zip(&learned.decisions) {
        assert_eq!((a.hit, a.admit, &a.evict), (b.hit, b.admit, &b.evict));
    }
}

#[test]
fn malformed_models_are_rejected() {
    let good = linear_as_mlp(&weights());
    assert!(MlpPolicy::new(good.clone()).is_ok());
    let mut wrong_width = good.clone();
    wrong_width.layers[0].weight[0].pop();
    assert!(MlpPolicy::new(wrong_width).is_err());
    let mut wrong_names = good.clone();
    wrong_names.feature_names[0] = "renamed".into();
    assert!(MlpPolicy::new(wrong_names).is_err());
    let mut zero_std = good.clone();
    zero_std.input_std[3] = 0.0;
    assert!(MlpPolicy::new(zero_std).is_err());
    let mut two_outputs = good.clone();
    two_outputs.layers[0].weight.push(vec![0.0; 9]);
    two_outputs.layers[0].bias.push(0.0);
    assert!(MlpPolicy::new(two_outputs).is_err());
    let mut rich = good;
    rich.features = FeatureSet::Rich;
    assert!(MlpPolicy::new(rich).is_err(), "basic-width model must not load as rich");
}

#[test]
fn hidden_layers_compute_relu_forward_pass() {
    let mut model = linear_as_mlp(&weights());
    model.layers = vec![
        Layer {
            weight: vec![vec![1.0; 9], vec![-1.0; 9]],
            bias: vec![0.0, 0.5],
            activation: Activation::Relu,
        },
        Layer {
            weight: vec![vec![2.0, 3.0]],
            bias: vec![1.0],
            activation: Activation::None,
        },
    ];
    model.validate().unwrap();
    let x = [0.1; 9];
    // hidden = [relu(0.9), relu(0.5 - 0.9)] = [0.9, 0]; out = 1 + 2 * 0.9
    assert!((model.forward(&x) - 2.8).abs() < 1e-12);
}

#[tokio::test]
async fn candidate_dump_has_one_chosen_row_per_eviction() {
    let (catalog, events) = trace("balanced");
    let mut dump = CandidateDump::new(Vec::new(), 8, 7, serde_json::json!({"trace": "balanced"}))
        .unwrap();
    let report = replay_observed(
        &catalog,
        &events,
        1,
        Some(Box::new(ClassicPolicy::lru())),
        Some(&mut dump),
    )
    .await
    .unwrap();
    let text = String::from_utf8(dump.into_inner()).unwrap();
    let mut lines = text.lines();
    let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(header["format"], "tvcache-candidates-v1");
    assert_eq!(header["feature_names"].as_array().unwrap().len(), RICH_NAMES.len());
    let rows: Vec<serde_json::Value> = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    let evictions = rows.iter().map(|r| r["eviction"].as_u64().unwrap()).max().unwrap() + 1;
    let chosen = rows.iter().filter(|r| r["chosen"] == true).count() as u64;
    assert_eq!(chosen, evictions);
    assert!(evictions > 0 && evictions <= report.evicted_entries);
    let mut per_eviction = std::collections::HashMap::new();
    for row in &rows {
        assert_eq!(row["x"].as_array().unwrap().len(), RICH_NAMES.len());
        let n = row["num_candidates"].as_u64().unwrap();
        per_eviction.entry(row["eviction"].as_u64().unwrap()).or_insert((0u64, n)).0 += 1;
    }
    assert!(per_eviction.values().all(|&(rows, n)| rows == 8.min(n)));
}

#[tokio::test]
async fn candidate_sampling_is_reproducible_and_stays_legal() {
    let (catalog, events) = trace("balanced");
    let mut model = linear_as_mlp(&weights());
    model.max_candidates = Some(4);
    let run = || {
        let policy = MlpPolicy::new(model.clone()).unwrap();
        replay(&catalog, &events, 1, Some(Box::new(policy)))
    };
    let (a, b) = (run().await.unwrap(), run().await.unwrap());
    check_decisions(&a, &catalog, &events).unwrap();
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.evicted_entries, b.evicted_entries);
    let mut zero = model.clone();
    zero.max_candidates = Some(0);
    assert!(MlpPolicy::new(zero).is_err());
}
