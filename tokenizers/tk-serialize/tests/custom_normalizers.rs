#![cfg(feature = "wordlevel")]

use std::{borrow::Cow, sync::Arc};
use tk_encode::pipeline::{EncodeOptions, Normalizer, PipelineNormalizer, PipelineTokenizer};

#[derive(Debug)]
struct Uppercase;
impl Normalizer for Uppercase {
    fn normalize<'a>(&self, input: &'a str, _: usize) -> tk_encode::Result<Cow<'a, str>> {
        Ok(Cow::Owned(input.to_uppercase()))
    }
}

fn tokenizer() -> PipelineTokenizer {
    let source = tk_convert::canonicalize_str(
        r#"{
        "normalizer":{"type":"Lowercase"},
        "pre_tokenizer":{"type":"WhitespaceSplit"},
        "model":{"type":"WordLevel","vocab":{"hello":0,"HELLO":1,"<unk>":2},"unk_token":"<unk>"}
    }"#,
    )
    .unwrap();
    tk_serialize::from_json(&source).unwrap()
}

#[test]
fn caller_normalization_retains_the_compiled_model() {
    let original = tokenizer();
    let model = original.get_model() as *const _;
    let tokenizer = original
        .map_normalizers(|_| PipelineNormalizer::Custom(Arc::new(Uppercase)))
        .unwrap();
    assert_eq!(model, tokenizer.get_model() as *const _);
    let encoded = tokenizer
        .encode("hello", &EncodeOptions::default())
        .wait()
        .unwrap();
    assert_eq!(encoded[0].ids()[0].id(), 1);
    let shared = tokenizer.clone();
    assert_eq!(model, shared.get_model() as *const _);
    assert!(shared.map_normalizers(|step| step).is_err());
    assert_eq!(
        tokenizer
            .encode("hello", &EncodeOptions::default())
            .wait()
            .unwrap()[0]
            .ids()[0]
            .id(),
        1
    );
}

#[cfg(feature = "serialize")]
#[test]
fn custom_normalizers_have_no_implicit_json_representation() {
    let tokenizer = tokenizer()
        .map_normalizers(|_| PipelineNormalizer::Custom(Arc::new(Uppercase)))
        .unwrap();
    assert!(
        tk_serialize::to_json(&tokenizer)
            .unwrap_err()
            .to_string()
            .contains("custom normalizers")
    );
}
