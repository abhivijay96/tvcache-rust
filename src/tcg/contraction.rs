use crate::tcg::{Node, ToolCallGraph};

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
}
