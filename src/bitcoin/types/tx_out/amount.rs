use std::{io::BufRead, io::Write};

use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

use crate::bitcoin::encoding::{Decodable, Encodable};

/// An amount.
///
/// The [`Amount`] type can be used to express Bitcoin amounts that support
/// arithmetic and conversion to various denominations.
#[derive(
    Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, BorshSerialize, BorshDeserialize,
)]
pub struct Amount(u64);

impl Amount {
    /// The zero amount.
    pub const ZERO: Amount = Amount(0);
    /// Exactly one satoshi.
    pub const ONE_SAT: Amount = Amount(1);
    /// Exactly one bitcoin.
    pub const ONE_BTC: Amount = Self::from_int_btc(1);
    /// The maximum value allowed as an amount. Useful for sanity checking.
    pub const MAX_MONEY: Amount = Self::from_int_btc(21_000_000);
    /// The minimum value of an amount.
    pub const MIN: Amount = Amount::ZERO;
    /// The maximum value of an amount.
    pub const MAX: Amount = Amount(u64::MAX);
    /// The number of bytes that an amount contributes to the size of a transaction.
    pub const SIZE: usize = 8; // Serialized length of a u64.

    /// Creates an [`Amount`] with satoshi precision and the given number of satoshis.
    pub const fn from_sat(satoshi: u64) -> Amount {
        Amount(satoshi)
    }

    /// Gets the number of satoshis in this [`Amount`].
    pub fn to_sat(self) -> u64 {
        self.0
    }

    /// Converts from a value expressing integer values of bitcoins to an [`Amount`]
    /// in const context.
    ///
    /// # Panics
    ///
    /// The function panics if the argument multiplied by the number of sats
    /// per bitcoin overflows a u64 type.
    pub const fn from_int_btc(btc: u64) -> Amount {
        match btc.checked_mul(100_000_000) {
            Some(amount) => Amount::from_sat(amount),
            None => panic!("checked_mul overflowed"),
        }
    }
}

impl Encodable for Amount {
    fn encode<W: Write + ?Sized>(&self, w: &mut W) -> Result<usize, std::io::Error> {
        self.0.encode(w)
    }
}

impl Decodable for Amount {
    fn decode_from_finite_reader<R: BufRead + ?Sized>(r: &mut R) -> Result<Self, std::io::Error> {
        let value = Decodable::decode_from_finite_reader(r)?;
        Ok(Amount::from_sat(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let amount = Amount::from_sat(1000);
        let mut buf = Vec::new();
        let size = amount.encode(&mut buf).unwrap();
        assert_eq!(size, Amount::SIZE);

        let decoded_amount = Amount::decode_from_finite_reader(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded_amount, amount);
    }
}
