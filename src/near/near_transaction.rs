use borsh::BorshSerialize;
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::{borsh, AccountId};

use super::types::{Action, PublicKey};

#[derive(Serialize, Deserialize, Debug, Clone, BorshSerialize)]
#[serde(crate = "near_sdk::serde")]
pub struct NearTransaction {
    /// An account on which behalf transaction is signed
    pub signer_id: AccountId,
    /// A public key of the access key which was used to sign an account.
    /// Access key holds permissions for calling certain kinds of actions.
    pub signer_public_key: PublicKey,
    /// Nonce is used to determine order of transaction in the pool.
    /// It increments for a combination of `signer_id` and `public_key`
    pub nonce: u64,
    /// Receiver account for this transaction
    pub receiver_id: AccountId,
    /// The hash of the block in the blockchain on top of which the given transaction is valid
    pub block_hash: [u8; 32],
    /// A list of actions to be applied
    pub actions: Vec<Action>,
}

impl NearTransaction {
    pub fn build_for_signing(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("failed to serialize NEAR transaction")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::near::types::{
        AccessKey as OmniAccessKey, AccessKeyPermission as OmniAccessKeyPermission,
        Action as OmniAction, AddKeyAction as OmniAddKeyAction,
        TransferAction as OmniTransferAction,
    };
    use crate::near::utils::PublicKeyStrExt;
    use near_crypto::{ED25519PublicKey, PublicKey};
    use near_primitives::{
        account::{AccessKey, AccessKeyPermission},
        action::{Action, AddKeyAction, TransferAction},
        hash::CryptoHash,
        transaction::TransactionV0,
    };

    #[derive(Debug)]
    struct TestCase {
        signer_id: &'static str,
        signer_public_key: &'static str,
        nonce: u64,
        receiver_id: &'static str,
        block_hash: &'static str,
        near_primitive_actions: Vec<Action>,
        omni_actions: Vec<OmniAction>,
    }

    fn create_test_cases() -> Vec<TestCase> {
        vec![
            // Simple transfer
            TestCase {
                signer_id: "alice.near",
                signer_public_key: "ed25519:6E8sCci9badyRkXb3JoRpBj5p8C6Tw41ELDZoiihKEtp",
                nonce: 1,
                receiver_id: "bob.near",
                block_hash: "4reLvkAWfqk5fsqio1KLudk46cqRz9erQdaHkWZKMJDZ",
                near_primitive_actions: vec![Action::Transfer(TransferAction { deposit: 1u128 })],
                omni_actions: vec![OmniAction::Transfer(OmniTransferAction { deposit: 1u128 })],
            },
            // Transfer and Add Key
            TestCase {
                signer_id: "forgetful-parent.testnet",
                signer_public_key: "ed25519:6E8sCci9badyRkXb3JoRpBj5p8C6Tw41ELDZoiihKEtp",
                nonce: 1,
                receiver_id: "forgetful-parent.testnet",
                block_hash: "4reLvkAWfqk5fsqio1KLudk46cqRz9erQdaHkWZKMJDZ",
                near_primitive_actions: vec![
                    Action::Transfer(TransferAction { deposit: 1u128 }),
                    Action::AddKey(Box::new(AddKeyAction {
                        public_key: PublicKey::ED25519(ED25519PublicKey(
                            "ed25519:6E8sCci9badyRkXb3JoRpBj5p8C6Tw41ELDZoiihKEtp"
                                .to_public_key_as_bytes()
                                .unwrap(),
                        )),
                        access_key: AccessKey {
                            nonce: 0,
                            permission: AccessKeyPermission::FullAccess,
                        },
                    })),
                ],
                omni_actions: vec![
                    OmniAction::Transfer(OmniTransferAction { deposit: 1u128 }),
                    OmniAction::AddKey(Box::new(OmniAddKeyAction {
                        public_key: "ed25519:6E8sCci9badyRkXb3JoRpBj5p8C6Tw41ELDZoiihKEtp"
                            .to_public_key()
                            .unwrap(),
                        access_key: OmniAccessKey {
                            nonce: 0,
                            permission: OmniAccessKeyPermission::FullAccess,
                        },
                    })),
                ],
            },
        ]
    }

    #[test]
    fn test_build_for_signing_for_near_against_near_primitives() {
        let test_cases = create_test_cases();

        for (i, test_case) in test_cases.iter().enumerate() {
            let near_primitive_v0_tx: TransactionV0 = TransactionV0 {
                signer_id: test_case.signer_id.parse().unwrap(),
                public_key: PublicKey::ED25519(ED25519PublicKey(
                    test_case
                        .signer_public_key
                        .to_public_key_as_bytes()
                        .unwrap(),
                )),
                nonce: test_case.nonce,
                receiver_id: test_case.receiver_id.parse().unwrap(),
                block_hash: CryptoHash(test_case.block_hash.to_fixed_32_bytes().unwrap()),
                actions: test_case.near_primitive_actions.clone(),
            };

            let serialized_near_primitive_v0_tx =
                borsh::to_vec(&near_primitive_v0_tx).expect("failed to serialize NEAR transaction");

            let omni_tx = NearTransaction {
                signer_id: test_case.signer_id.parse().unwrap(),
                signer_public_key: test_case.signer_public_key.to_public_key().unwrap(),
                nonce: test_case.nonce,
                receiver_id: test_case.receiver_id.parse().unwrap(),
                block_hash: test_case.block_hash.to_fixed_32_bytes().unwrap(),
                actions: test_case.omni_actions.clone(),
            };

            let serialized_omni_tx = omni_tx.build_for_signing();

            assert_eq!(
                serialized_near_primitive_v0_tx, serialized_omni_tx,
                "Test case {} failed: serialized transactions do not match",
                i
            );
        }
    }
}
