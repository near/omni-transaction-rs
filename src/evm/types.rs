//! Types used by the EVM transaction builder.
use near_sdk::serde::{Deserialize, Serialize};
use schemars::JsonSchema;

pub type Address = [u8; 20];

pub type AccessList = Vec<(Address, Vec<[u8; 32]>)>;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(crate = "near_sdk::serde")]
pub struct Signature {
    pub v: u64,
    pub r: Vec<u8>,
    pub s: Vec<u8>,
}
