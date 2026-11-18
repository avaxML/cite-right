use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

pub(crate) fn canonical_token_id(token: u32, alias_ids: Option<&HashMap<u32, u32>>) -> u32 {
    alias_ids
        .and_then(|aliases| aliases.get(&token).copied())
        .unwrap_or(token)
}

pub(crate) fn remap_token_ids<'a>(
    tokens: &'a [u32],
    alias_ids: Option<&HashMap<u32, u32>>,
) -> Cow<'a, [u32]> {
    if alias_ids.is_some_and(|aliases| !aliases.is_empty()) {
        Cow::Owned(
            tokens
                .iter()
                .map(|&token| canonical_token_id(token, alias_ids))
                .collect(),
        )
    } else {
        Cow::Borrowed(tokens)
    }
}

/// Alias IDs resolved once for a single public alignment call.
#[pyclass(frozen)]
pub struct AliasRemap {
    pub(crate) ids: HashMap<u32, u32>,
}

#[pymethods]
impl AliasRemap {
    #[new]
    fn new(ids: HashMap<u32, u32>) -> Self {
        Self { ids }
    }
}
