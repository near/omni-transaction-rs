use super::{height::Height, time::Time};
use borsh::{BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};

/// Locktime itself is an unsigned 4-byte integer which can be parsed two ways:
///
/// If less than 500 million, locktime is parsed as a block height.
/// The transaction can be added to any block which has this height or higher.
///
/// If greater than or equal to 500 million, locktime is parsed using the Unix epoch time format
/// (the number of seconds elapsed since 1970-01-01T00:00 UTC—currently over 1.395 billion).
/// The transaction can be added to any block whose block time is greater than the locktime.
///
/// [Bitcoin Devguide]: https://developer.bitcoin.org/devguide/transactions.html#locktime-and-sequence-number
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, BorshSerialize, BorshDeserialize,
)]
pub struct LockTime(u32);

impl LockTime {
    pub fn from_height(height: u32) -> Result<Self, String> {
        if Height::is_valid(height) {
            Ok(LockTime(height))
        } else {
            Err(format!("Invalid block height: {}", height))
        }
    }

    pub fn from_time(time: u32) -> Result<Self, String> {
        if Time::is_valid(time) {
            Ok(LockTime(time))
        } else {
            Err(format!("Invalid timestamp: {}", time))
        }
    }

    pub fn is_block_height(&self) -> bool {
        Height::is_valid(self.0)
    }

    pub fn is_unix_time(&self) -> bool {
        Time::is_valid(self.0)
    }

    pub fn to_u32(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitcoin::types::Height;

    #[test]
    fn test_locktime_from_height() {
        let h = 100;
        let height = LockTime::from_height(h).unwrap();

        assert!(height.is_block_height());
        assert!(!height.is_unix_time());
        assert_eq!(height.to_u32(), h);
    }

    #[test]
    fn test_locktime_from_time() {
        let time = LockTime::from_time(Time::MIN + 100).unwrap();

        assert!(!time.is_block_height());
        assert!(time.is_unix_time());
        assert_eq!(time.to_u32(), Time::MIN + 100);
    }

    #[test]
    fn test_locktime_invalid_height() {
        assert!(LockTime::from_height(Height::MAX + 1).is_err());
    }

    #[test]
    fn test_locktime_invalid_time() {
        assert!(LockTime::from_time(Time::MIN - 1).is_err());
    }

    #[test]
    fn test_locktime_serialization() {
        let locktime = LockTime::from_height(100).unwrap();
        let serialized = serde_json::to_string(&locktime).unwrap();
        let deserialized: LockTime = serde_json::from_str(&serialized).unwrap();
        assert_eq!(locktime, deserialized);
    }
}
