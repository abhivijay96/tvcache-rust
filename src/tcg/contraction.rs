use crate::tcg::{Node, ToolCallGraph};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// A child with at least one similar sibling it may still need comparing with.
#[derive(Debug)]
pub struct Sibling {
    pub keys: Vec<String>,
    pub settled: bool,
    pub has_snapshot: bool,
}

/// Similar siblings under one parent. Each pair of similar siblings
/// indexes into `children`.
#[derive(Debug)]
pub struct SiblingGroup {
    pub parent_path: Vec<String>,
    pub children: Vec<Sibling>,
    pub pairs: Vec<(usize, usize, f32)>,
}

impl ToolCallGraph {
    /// Returns whether any node is deeper than depth, and the similar siblings at depth.
    pub fn sibling_groups(&self, depth: usize, threshold: f32) -> (bool, Vec<SiblingGroup>) {
        let Some(parent_depth) = depth.checked_sub(1) else {
            return (!self.root.children.is_empty(), Vec::new());
        };
        let mut parents = Vec::new();
        collect_parents(&self.root, parent_depth, &mut Vec::new(), &mut parents);

        let deeper_nodes_exist = parents.iter().any(|(_, parent)| {
            parent
                .iter_children()
                .any(|(_, child)| !child.children.is_empty())
        });
        let groups = parents
            .into_iter()
            .filter_map(|(path, parent)| sibling_group(path, parent, threshold))
            .collect();
        (deeper_nodes_exist, groups)
    }
}

fn collect_parents<'a>(
    node: &'a Node,
    remaining_depth: usize,
    path: &mut Vec<String>,
    parents: &mut Vec<(Vec<String>, &'a Node)>,
) {
    if remaining_depth == 0 {
        parents.push((path.clone(), node));
        return;
    }
    for (key, child) in node.iter_children() {
        path.push(key.clone());
        collect_parents(child, remaining_depth - 1, path, parents);
        path.pop();
    }
}

/// Returns None if no pair of the parent's children qualifies.
fn sibling_group(parent_path: Vec<String>, parent: &Node, threshold: f32) -> Option<SiblingGroup> {
    let siblings: Vec<&Node> = parent.iter_children().map(|(_, child)| child).collect();
    let mut pairs = Vec::new();
    for i in 0..siblings.len() {
        for j in i + 1..siblings.len() {
            // Two blue siblings have already been compared.
            if siblings[i].settled && siblings[j].settled {
                continue;
            }
            if let Some(score) = similarity(siblings[i], siblings[j]) {
                if score >= threshold {
                    pairs.push((i, j, score));
                }
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }

    let mut members: Vec<usize> = pairs.iter().flat_map(|&(i, j, _)| [i, j]).collect();
    members.sort_unstable();
    members.dedup();
    let index = |position: usize| {
        members
            .binary_search(&position)
            .expect("Pair member missing")
    };
    Some(SiblingGroup {
        parent_path,
        children: members
            .iter()
            .map(|&position| sibling(siblings[position]))
            .collect(),
        pairs: pairs
            .iter()
            .map(|&(i, j, score)| (index(i), index(j), score))
            .collect(),
    })
}

fn sibling(node: &Node) -> Sibling {
    let mut keys: Vec<String> = node.tool_calls.keys().cloned().collect();
    keys.sort();
    Sibling {
        keys,
        settled: node.settled,
        has_snapshot: node.sandbox_id.is_some(),
    }
}

/// What a merge removed, for the graph's counters.
#[derive(Debug, Default)]
pub struct Merged {
    pub absorbed_nodes: u32,
    pub reclaimable_sandboxes: Vec<String>,
}

/// Absorbs b into a, recursively merging children that share a tool-call key.
fn merge(a: &mut Node, b: Node, merged: &mut Merged) {
    merged.absorbed_nodes += 1;
    a.settled = false;
    // On a key clash, keep a's entry but fill in what only b has.
    for (key, entry) in b.tool_calls {
        let kept = a.tool_calls.entry(key).or_default();
        if kept.value.is_none() {
            (kept.value, kept.runtime) = (entry.value, entry.runtime);
        }
        kept.embedding = kept.embedding.take().or(entry.embedding);
    }
    if let Some(sandbox_id) = b.sandbox_id {
        if a.sandbox_id.is_none() {
            a.sandbox_id = Some(sandbox_id);
        } else {
            merged.reclaimable_sandboxes.push(sandbox_id);
        }
    }

    a.number_of_cache_hits += b.number_of_cache_hits;
    a.number_of_lpm_hits += b.number_of_lpm_hits;
    // A node without a TTL never expires, and otherwise the later expiry wins.
    let a_expiry = a.inserted_at_seconds + a.ttl as u64;
    if !b.ttl_present {
        a.ttl_present = false;
    } else if a.ttl_present && b.inserted_at_seconds + b.ttl as u64 > a_expiry {
        a.inserted_at_seconds = b.inserted_at_seconds;
        a.ttl = b.ttl;
    }

    let a_count = a.children.len();
    let children: Vec<Node> = std::mem::take(&mut a.children)
        .into_iter()
        .chain(b.children)
        .collect();
    let components = key_components(&children);
    let mut children: Vec<Option<Node>> = children.into_iter().map(Some).collect();
    for component in components {
        let mut survivor = children[component[0]].take().expect("Child already merged");
        if component.len() > 1 {
            // Siblings never share keys, so every multi-child component includes one of a's.
            debug_assert!(component[0] < a_count);
            for &index in &component[1..] {
                let child = children[index].take().expect("Child already merged");
                merge(&mut survivor, child, merged);
            }
        } else if component[0] >= a_count {
            survivor.settled = false;
        }
        a.children.push(survivor);
    }
    a.update_size_in_bytes();
}

/// Groups nodes whose tool-call maps share a key, directly or through other nodes.
/// Each component lists its indexes in ascending order, and components are
/// ordered by their first index.
fn key_components(nodes: &[Node]) -> Vec<Vec<usize>> {
    fn find(roots: &mut [usize], mut index: usize) -> usize {
        while roots[index] != index {
            roots[index] = roots[roots[index]];
            index = roots[index];
        }
        index
    }

    // Each set's root is its smallest index.
    let mut roots: Vec<usize> = (0..nodes.len()).collect();
    let mut owners: HashMap<&str, usize> = HashMap::new();
    for (index, node) in nodes.iter().enumerate() {
        for key in node.tool_calls.keys() {
            match owners.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(index);
                }
                Entry::Occupied(entry) => {
                    let (a, b) = (find(&mut roots, index), find(&mut roots, *entry.get()));
                    roots[a.max(b)] = a.min(b);
                }
            }
        }
    }

    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut component_of_root = vec![0; nodes.len()];
    for index in 0..nodes.len() {
        let root = find(&mut roots, index);
        if root == index {
            component_of_root[index] = components.len();
            components.push(vec![index]);
        } else {
            components[component_of_root[root]].push(index);
        }
    }
    components
}

fn embeddings(node: &Node) -> impl Iterator<Item = &[f32]> {
    node.tool_calls
        .values()
        .filter_map(|entry| entry.embedding.as_deref())
}

/// Returns the highest cosine similarity over pairs of the nodes' entry embeddings,
/// or None if no pair is valid.
fn similarity(a: &Node, b: &Node) -> Option<f32> {
    embeddings(a)
        .flat_map(|x| embeddings(b).filter_map(move |y| cosine(x, y)))
        .reduce(f32::max)
}

/// Returns None for mismatched dimensions, zero vectors and non-finite input.
fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let score = dot / (norm_a * norm_b).sqrt();
    score.is_finite().then_some(score as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcg::node::ToolCallEntry;

    fn node(entries: &[Option<Vec<f32>>]) -> Node {
        let mut node = Node::default();
        for (index, embedding) in entries.iter().enumerate() {
            let entry = ToolCallEntry {
                embedding: embedding.clone(),
                ..Default::default()
            };
            node.tool_calls.insert(index.to_string(), entry);
        }
        node
    }

    fn score(a: &Node, b: &Node) -> Option<i32> {
        similarity(a, b).map(|score| (score * 100.0).round() as i32)
    }

    #[test]
    fn similarity_takes_the_closest_valid_entry_pair() {
        let a = node(&[Some(vec![1.0, 0.0])]);
        assert_eq!(similarity(&a, &node(&[Some(vec![2.0, 0.0])])), Some(1.0));
        assert_eq!(score(&a, &node(&[Some(vec![0.8, 0.6])])), Some(80));
        assert_eq!(score(&a, &node(&[Some(vec![-1.0, 0.0])])), Some(-100));

        // Merged nodes score by their closest pair of entries.
        let merged = node(&[Some(vec![0.0, 1.0]), Some(vec![0.8, 0.6]), None]);
        assert_eq!(score(&a, &merged), Some(80));
        assert_eq!(similarity(&a, &merged), similarity(&merged, &a));
        assert_eq!(score(&node(&[Some(vec![0.0, 2.0])]), &merged), Some(100));

        // Zero vectors, mismatched dimensions and missing embeddings are skipped.
        assert_eq!(
            score(&a, &node(&[Some(vec![0.0, 0.0]), Some(vec![3.0, 0.0])])),
            Some(100)
        );
        assert_eq!(score(&a, &node(&[Some(vec![0.0, 0.0])])), None);
        assert_eq!(score(&a, &node(&[Some(vec![1.0, 0.0, 0.0])])), None);
        assert_eq!(score(&a, &node(&[None])), None);
        assert_eq!(score(&a, &Node::default()), None);
        let nan = node(&[Some(vec![f32::NAN, 0.0]), Some(vec![1.0, 0.0])]);
        assert_eq!(score(&a, &nan), Some(100));
        assert_eq!(score(&a, &node(&[Some(vec![f32::INFINITY, 0.0])])), None);
    }

    fn insert(graph: &mut ToolCallGraph, path: &[&str], embedding: Option<Vec<f32>>) {
        let path = path.iter().map(|key| key.to_string()).collect();
        graph.insert(path, None, None, 1, None, false, embedding);
    }

    fn summary(group: &SiblingGroup) -> (Vec<String>, Vec<(usize, usize, i32)>) {
        let keys = group.children.iter().map(|child| child.keys.join("+"));
        let percent = |score: f32| (score * 100.0).round() as i32;
        let pairs = group
            .pairs
            .iter()
            .map(|&(i, j, score)| (i, j, percent(score)));
        (keys.collect(), pairs.collect())
    }

    #[test]
    fn sibling_group_follows_qualifying_rules() {
        let mut graph = ToolCallGraph::new();
        assert!(sibling_group(Vec::new(), &graph.root, 0.0).is_none());
        insert(&mut graph, &["a"], Some(vec![1.0, 0.0]));
        assert!(sibling_group(Vec::new(), &graph.root, 0.0).is_none());
        insert(&mut graph, &["b"], Some(vec![0.8, 0.6]));
        insert(&mut graph, &["c"], Some(vec![0.0, 1.0]));
        insert(&mut graph, &["missing"], None);
        graph.root.child_mut("b").unwrap().sandbox_id = Some("snapshot".into());

        // The b-c pair scores exactly at the threshold, which qualifies.
        let group = sibling_group(vec!["parent".into()], &graph.root, 0.6).unwrap();
        assert_eq!(group.parent_path, ["parent"]);
        let expected = vec!["a".to_string(), "b".into(), "c".into()];
        assert_eq!(summary(&group), (expected, vec![(0, 1, 80), (1, 2, 60)]));
        assert!(!group.children[0].has_snapshot && group.children[1].has_snapshot);
        assert!(sibling_group(Vec::new(), &graph.root, 0.81).is_none());

        // Blue pairs are skipped, and only children in some pair are listed.
        for key in ["a", "b"] {
            graph.root.child_mut(key).unwrap().settled = true;
        }
        let group = sibling_group(Vec::new(), &graph.root, 0.6).unwrap();
        let expected = vec!["b".to_string(), "c".into()];
        assert_eq!(summary(&group), (expected, vec![(0, 1, 60)]));
        assert!(group.children[0].settled && !group.children[1].settled);

        // A merged node lists every key and pairs through its closest embedded entry.
        let alias = ToolCallEntry {
            embedding: Some(vec![1.0, 0.0]),
            ..Default::default()
        };
        let c = graph.root.child_mut("c").unwrap();
        c.tool_calls.insert("alias".into(), alias);
        c.tool_calls
            .insert("empty".into(), ToolCallEntry::default());
        let group = sibling_group(Vec::new(), &graph.root, 0.6).unwrap();
        let expected = vec!["a".to_string(), "b".into(), "alias+c+empty".into()];
        assert_eq!(summary(&group), (expected, vec![(0, 2, 100), (1, 2, 80)]));
    }

    #[test]
    fn sibling_groups_report_parents_and_deeper_nodes() {
        let mut graph = ToolCallGraph::new();
        assert!(!graph.sibling_groups(0, 0.5).0);
        insert(&mut graph, &["a"], None);
        insert(&mut graph, &["b"], None);
        insert(&mut graph, &["a", "x"], Some(vec![1.0, 0.0]));
        insert(&mut graph, &["a", "y"], Some(vec![0.0, 1.0]));
        insert(&mut graph, &["b", "z"], Some(vec![1.0, 0.0]));
        insert(&mut graph, &["b", "v"], Some(vec![1.0, 0.1]));
        insert(&mut graph, &["a", "x", "p"], Some(vec![1.0, 0.0]));
        insert(&mut graph, &["a", "x", "q"], Some(vec![1.0, 0.0]));

        // A depth without pairs can still have deeper nodes.
        let (deeper, groups) = graph.sibling_groups(0, 0.5);
        assert!(deeper && groups.is_empty());
        let (deeper, groups) = graph.sibling_groups(1, 0.5);
        assert!(deeper && groups.is_empty());

        // Parents without pairs are omitted, and cousins x and z are never paired.
        let (deeper, groups) = graph.sibling_groups(2, 0.5);
        assert!(deeper && groups.len() == 1 && groups[0].parent_path == ["b"]);
        assert_eq!(summary(&groups[0]).0, ["z", "v"]);

        let (deeper, groups) = graph.sibling_groups(3, 0.5);
        assert!(!deeper && groups.len() == 1 && groups[0].parent_path == ["a", "x"]);
        let (deeper, groups) = graph.sibling_groups(9, 0.5);
        assert!(!deeper && groups.is_empty());
    }

    #[test]
    fn key_components_join_nodes_through_shared_keys() {
        let node = |keys: &[&str]| {
            let mut node = Node::default();
            for key in keys {
                node.tool_calls
                    .insert(key.to_string(), ToolCallEntry::default());
            }
            node
        };
        assert!(key_components(&[]).is_empty());

        // b2 bridges a2 and a3, and b4 reaches a1 only through b3. Fresh maps
        // vary key order, which decides the direction of each union.
        for _ in 0..20 {
            let nodes = [
                node(&["k1"]),
                node(&["k2"]),
                node(&["k3"]),
                node(&["k5"]),
                node(&["k3", "k2"]),
                node(&["k4", "k1"]),
                node(&["k6", "k4"]),
                node(&["k7"]),
            ];
            let components = key_components(&nodes);
            assert_eq!(components, [vec![0, 5, 6], vec![1, 2, 4], vec![3], vec![7]]);
        }
    }

    fn tree(keys: &[&str], settled: bool, children: Vec<Node>) -> Node {
        let mut node = Node::default();
        for key in keys {
            let entry = ToolCallEntry {
                value: Some(key.to_string()),
                ..Default::default()
            };
            node.tool_calls.insert(key.to_string(), entry);
        }
        node.settled = settled;
        node.children = children;
        node
    }

    fn leaf(keys: &[&str]) -> Node {
        tree(keys, true, Vec::new())
    }

    #[test]
    fn merge_combines_entries_snapshots_and_stats() {
        let mut a = leaf(&["k", "both"]);
        let mut b = leaf(&["k", "both", "alias"]);
        a.tool_calls.get_mut("k").unwrap().value = None;
        let k = b.tool_calls.get_mut("k").unwrap();
        (k.value, k.runtime, k.embedding) = (Some("b".into()), 7, Some(vec![1.0]));
        a.tool_calls.get_mut("both").unwrap().embedding = Some(vec![2.0]);
        let both = b.tool_calls.get_mut("both").unwrap();
        (both.value, both.runtime, both.embedding) = (Some("b".into()), 9, Some(vec![3.0]));
        (a.sandbox_id, b.sandbox_id) = (Some("sa".into()), Some("sb".into()));
        (a.number_of_cache_hits, a.number_of_lpm_hits) = (2, 1);
        (b.number_of_cache_hits, b.number_of_lpm_hits) = (3, 4);

        let mut merged = Merged::default();
        merge(&mut a, b, &mut merged);
        let k = &a.tool_calls["k"];
        assert!(
            k.value.as_deref() == Some("b") && k.runtime == 7 && k.embedding == Some(vec![1.0])
        );
        let both = &a.tool_calls["both"];
        assert!(both.value.as_deref() == Some("both") && both.runtime == 0);
        assert_eq!(both.embedding, Some(vec![2.0]));
        assert!(
            !a.settled
                && a.tool_calls.contains_key("alias")
                && a.sandbox_id.as_deref() == Some("sa")
        );
        assert_eq!(
            (merged.absorbed_nodes, merged.reclaimable_sandboxes),
            (1, vec!["sb".into()])
        );
        assert_eq!((a.number_of_cache_hits, a.number_of_lpm_hits), (5, 5));

        // A missing snapshot is taken from b.
        let mut b = tree(&["k"], false, Vec::new());
        (a.sandbox_id, b.sandbox_id) = (None, Some("sb".into()));
        let mut merged = Merged::default();
        merge(&mut a, b, &mut merged);
        assert!(a.sandbox_id.as_deref() == Some("sb") && merged.reclaimable_sandboxes.is_empty());
    }

    #[test]
    fn merge_keeps_the_later_expiry() {
        // Each TTL is (present, ttl, inserted_at_seconds).
        for (a_ttl, b_ttl, expected) in [
            ((true, 10, 100), (true, 5, 120), (true, 5, 120)),
            ((true, 10, 100), (true, 5, 100), (true, 10, 100)),
            ((true, 10, 100), (true, 5, 103), (true, 10, 100)),
            ((false, 0, 100), (true, 5, 120), (false, 0, 100)),
            ((true, 10, 100), (false, 0, 120), (false, 10, 100)),
        ] {
            let (mut a, mut b) = (leaf(&["k"]), leaf(&["k"]));
            (a.ttl_present, a.ttl, a.inserted_at_seconds) = a_ttl;
            (b.ttl_present, b.ttl, b.inserted_at_seconds) = b_ttl;
            merge(&mut a, b, &mut Merged::default());
            assert_eq!((a.ttl_present, a.ttl, a.inserted_at_seconds), expected);
        }
    }

    #[test]
    fn merge_recurses_into_children_with_shared_keys() {
        let mut a_x = tree(&["x"], true, vec![leaf(&["g"])]);
        a_x.sandbox_id = Some("sa".into());
        let a_children = vec![a_x, leaf(&["y"]), leaf(&["z"]), leaf(&["q"])];
        let mut b_x = tree(&["x"], true, vec![leaf(&["g"]), leaf(&["h"])]);
        b_x.sandbox_id = Some("sb".into());
        let b_children = vec![b_x, leaf(&["y", "z"]), leaf(&["n"])];
        let mut a = tree(&["a"], true, a_children);

        let mut merged = Merged::default();
        merge(&mut a, tree(&["b"], true, b_children), &mut merged);
        let summary = |node: &Node| {
            let mut keys: Vec<&str> = node.tool_calls.keys().map(String::as_str).collect();
            keys.sort();
            (keys.join("+"), node.settled)
        };
        let children: Vec<_> = a.children.iter().map(summary).collect();
        let expected = [("x", false), ("y+z", false), ("q", true), ("n", false)];
        assert_eq!(
            children,
            expected.map(|(keys, settled)| (keys.to_string(), settled))
        );
        let grandchildren: Vec<_> = a.children[0].children.iter().map(summary).collect();
        assert_eq!(
            grandchildren,
            [("g".to_string(), false), ("h".to_string(), false)]
        );
        assert_eq!(
            (merged.absorbed_nodes, merged.reclaimable_sandboxes),
            (5, vec!["sb".into()])
        );
        assert_eq!(a.walk_subtcg(0, &mut |count, _| count + 1), 7);
    }
}
