//! Move identifier and module id types.
use core::fmt;

use super::address::AccountAddress;
use crate::bcs_encoding::{BcsEncode, BcsWriter};
#[cfg(feature = "schemars")]
use schemars::JsonSchema;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A valid Move identifier (module or function name).
///
/// Identifiers match `[a-zA-Z][a-zA-Z0-9_]*` or `_[a-zA-Z0-9_]+` (a lone `_`
/// is not a valid identifier), mirroring `move-core-types`'
/// `identifier::is_valid`.
///
/// In BCS an identifier is a string: ULEB128 byte length followed by the
/// UTF-8 bytes. The JSON (serde) representation is a plain string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Identifier(String);

impl Identifier {
    /// Creates an identifier, validating the Move identifier charset.
    ///
    /// # Errors
    ///
    /// Returns [`IdentifierParseError`] if the string is not a valid Move
    /// identifier.
    pub fn new(name: impl Into<String>) -> Result<Self, IdentifierParseError> {
        let name = name.into();
        if is_valid_identifier(&name) {
            Ok(Self(name))
        } else {
            Err(IdentifierParseError)
        }
    }

    /// Returns the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the identifier, returning the underlying string.
    pub fn into_string(self) -> String {
        self.0
    }
}

/// Returns `true` if `name` matches `[a-zA-Z][a-zA-Z0-9_]*` or `_[a-zA-Z0-9_]+`.
fn is_valid_identifier(name: &str) -> bool {
    let bytes = name.as_bytes();
    let rest = match bytes.first() {
        Some(b'a'..=b'z' | b'A'..=b'Z') => &bytes[1..],
        Some(b'_') if bytes.len() > 1 => &bytes[1..],
        _ => return false,
    };
    rest.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl core::str::FromStr for Identifier {
    type Err = IdentifierParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl From<Identifier> for String {
    fn from(identifier: Identifier) -> Self {
        identifier.0
    }
}

impl BcsEncode for Identifier {
    fn bcs_encode(&self, writer: &mut BcsWriter) {
        writer.write_string(&self.0);
    }
}

/// Error returned when a string is not a valid Move identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentifierParseError;

impl fmt::Display for IdentifierParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid Move identifier: expected [a-zA-Z][a-zA-Z0-9_]* or _[a-zA-Z0-9_]+"
        )
    }
}

impl std::error::Error for IdentifierParseError {}

// Serde is implemented by hand so that deserialization enforces the Move
// identifier invariant (a derive on the private field would bypass `new`).
#[cfg(feature = "serde")]
impl Serialize for Identifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Identifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::new(name).map_err(serde::de::Error::custom)
    }
}

// The schema mirrors the serde form (a plain string).
#[cfg(feature = "schemars")]
impl JsonSchema for Identifier {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Identifier".into()
    }

    fn json_schema(generator: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
        <String>::json_schema(generator)
    }
}

/// The id of a Move module: the address that published it plus its name.
///
/// In BCS: 32 raw address bytes followed by the name as a ULEB128
/// length-prefixed string (see `move-core-types`' `language_storage.rs`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct ModuleId {
    /// The address that published the module.
    pub address: AccountAddress,
    /// The module name.
    pub name: Identifier,
}

impl ModuleId {
    /// Creates a module id from an address and a module name.
    pub const fn new(address: AccountAddress, name: Identifier) -> Self {
        Self { address, name }
    }
}

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.address, self.name)
    }
}

impl BcsEncode for ModuleId {
    fn bcs_encode(&self, writer: &mut BcsWriter) {
        self.address.bcs_encode(writer);
        self.name.bcs_encode(writer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bcs_encoding::to_bytes;

    #[test]
    fn test_identifier_validation() {
        assert!(Identifier::new("transfer").is_ok());
        assert!(Identifier::new("aptos_coin").is_ok());
        assert!(Identifier::new("_private").is_ok());
        assert!(Identifier::new("v2").is_ok());
        assert!(Identifier::new("").is_err());
        assert!(Identifier::new("_").is_err());
        assert!(Identifier::new("1abc").is_err());
        assert!(Identifier::new("has-dash").is_err());
        assert!(Identifier::new("has space").is_err());
        assert!(Identifier::new("emoji🚀").is_err());
    }

    #[test]
    fn test_identifier_bcs_matches_reference_bcs_string() {
        let identifier = Identifier::new("aptos_coin").unwrap();
        let encoded = to_bytes(&identifier);
        assert_eq!(encoded, bcs::to_bytes(&"aptos_coin").unwrap());
        assert_eq!(hex::encode(&encoded), "0a6170746f735f636f696e");
    }

    #[test]
    fn test_module_id_bcs_is_address_then_name() {
        let module_id = ModuleId::new(
            AccountAddress::from_hex("0x1222").unwrap(),
            Identifier::new("aptos_coin").unwrap(),
        );
        let encoded = to_bytes(&module_id);
        assert_eq!(
            hex::encode(encoded),
            "00000000000000000000000000000000000000000000000000000000000012220a6170746f735f636f696e"
        );
    }

    #[test]
    #[cfg(feature = "serde_json")]
    fn test_identifier_serde_round_trip_enforces_validation() {
        let identifier = Identifier::new("transfer").unwrap();
        let json = serde_json::to_string(&identifier).unwrap();
        assert_eq!(json, "\"transfer\"");
        let back: Identifier = serde_json::from_str(&json).unwrap();
        assert_eq!(back, identifier);

        let invalid: Result<Identifier, _> = serde_json::from_str("\"not valid\"");
        assert!(invalid.is_err());
    }
}
