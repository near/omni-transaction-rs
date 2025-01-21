//! Low level transaction builders for different blockchains.
#[cfg(feature = "bitcoin")]
use crate::bitcoin::BitcoinTransactionBuilder;

#[cfg(feature = "evm")]
use crate::evm::EVMTransactionBuilder;

#[cfg(feature = "near")]
use crate::near::NearTransactionBuilder;

#[cfg(feature = "near")]
pub type NEAR = NearTransactionBuilder;

#[cfg(feature = "evm")]
pub type EVM = EVMTransactionBuilder;

#[cfg(feature = "bitcoin")]
pub type BITCOIN = BitcoinTransactionBuilder;
