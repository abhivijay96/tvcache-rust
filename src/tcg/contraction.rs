use crate::tcg::Node;

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
}
