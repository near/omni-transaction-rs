use bitcoin::absolute::Height;
use bitcoin::bip32::{ChildNumber, DerivationPath, Xpub};
use bitcoin::locktime::absolute;
use bitcoin::script::{Builder, Instruction};
use bitcoin::secp256k1::{Message, Secp256k1};
use bitcoin::sighash::{EcdsaSighashType, SighashCache};
use bitcoin::{
    transaction, Address, Amount, Network, OutPoint, ScriptBuf, Sequence, Transaction, TxIn, TxOut,
    Witness,
};
use bitcoind::Conf;

use eyre::Result;
use serde_json::json;
use std::result::Result::Ok;

const SPEND_AMOUNT: Amount = Amount::from_sat(500_000_000);

mod utils;

pub use utils::bitcoin_utils::*;

#[tokio::test]
async fn test_send_p2pkh_using_rust_bitcoin() -> Result<()> {
    let conf = Conf::default();
    let bitcoind = bitcoind::BitcoinD::from_downloaded_with_conf(&conf).unwrap();
    let client: &bitcoind::Client = &bitcoind.client;

    // Setup Bob and Alice addresses
    let (bob_address, bob_script_pubkey) =
        get_address_info_for(client, "Bob").expect("Failed to get address info for Bob");

    let (alice_address, alice_script_pubkey) =
        get_address_info_for(client, "Alice").expect("Failed to get address info for Alice");

    // Get descriptors
    let master_key = get_master_key_of_regtest_node(client).expect("Failed to get master key");

    // Initialize secp256k1 context
    let secp = Secp256k1::new();

    // Derive child private key using path m/44h/1h/0h
    let path = "m/44h/1h/0h".parse::<DerivationPath>().unwrap();
    let child = master_key.derive_priv(&secp, &path).unwrap();
    println!("Child at {}: {}", path, child);

    let xpub = Xpub::from_priv(&secp, &child);
    println!("Public key at {}: {}", path, xpub);

    // Generate first P2PKH address at m/0/0
    let zero = ChildNumber::Normal { index: 0 };
    let public_key = xpub.derive_pub(&secp, &[zero, zero]).unwrap().public_key;
    let bitcoin_public_key = bitcoin::PublicKey::new(public_key);
    let derived_bob_address = Address::p2pkh(&bitcoin_public_key, Network::Regtest);

    assert_eq!(bob_address, derived_bob_address);

    // Get private key for first P2PKH address
    let first_priv_key = child
        .derive_priv(&secp, &DerivationPath::from(vec![zero, zero]))
        .unwrap()
        .private_key;

    println!(
        "Private key for first receiving address: {:?}",
        first_priv_key
    );

    // Generate 101 blocks to the address
    client.generate_to_address(101, &bob_address)?;

    // List UTXOs for Bob
    let min_conf = 1;
    let max_conf = 9999999;
    let include_unsafe = true;
    let query_options = json!({});
    let unspent_utxos_bob: Vec<serde_json::Value> = client.call(
        "listunspent",
        &[
            json!(min_conf),
            json!(max_conf),
            json!(vec![bob_address.to_string()]),
            json!(include_unsafe),
            query_options.clone(),
        ],
    )?;

    println!("UTXOs for Bob: {:?}", unspent_utxos_bob);

    // Get the first UTXO
    let first_unspent = unspent_utxos_bob
        .into_iter()
        .next()
        .expect("There should be at least one unspent output");

    println!("First UTXO: {:?}", first_unspent);

    // Verify UTXO belongs to our address and has the correct amount
    assert_eq!(
        first_unspent["address"].as_str().unwrap(),
        bob_address.to_string(),
        "UTXO doesn't belong to Bob"
    );

    let utxo_amount = Amount::from_btc(first_unspent["amount"].as_f64().unwrap())?;
    println!("UTXO amount: {:?}", utxo_amount);

    // Check if the UTXO amount is 50 BTC
    assert_eq!(
        utxo_amount.to_sat(),
        5_000_000_000,
        "UTXO amount is not 50 BTC"
    );

    // Generate second (alice) P2PKH address at m/0/1
    let one = ChildNumber::Normal { index: 1 };
    let alice_public_key = xpub.derive_pub(&secp, &[zero, one]).unwrap().public_key;
    let alice_bitcoin_public_key = bitcoin::PublicKey::new(alice_public_key);
    let derived_alice_address = Address::p2pkh(&alice_bitcoin_public_key, Network::Regtest);
    println!("Alice derived address: {}", derived_alice_address);

    assert_eq!(alice_address, derived_alice_address);

    let txin = TxIn {
        previous_output: OutPoint::new(
            first_unspent["txid"].as_str().unwrap().parse()?,
            first_unspent["vout"].as_u64().unwrap() as u32,
        ),
        script_sig: ScriptBuf::new(), // Initially empty, will be filled later with the signature
        sequence: Sequence::MAX,
        witness: Witness::default(),
    };

    let txout = TxOut {
        value: SPEND_AMOUNT,
        script_pubkey: alice_script_pubkey.clone(),
    };

    println!("txout: {:?}", txout);

    let change_amount: Amount = utxo_amount - SPEND_AMOUNT - Amount::from_sat(1000); // 1000 satoshis for fee
    println!("change_amount: {:?}", change_amount);

    let change_txout = TxOut {
        value: change_amount,
        script_pubkey: bob_script_pubkey.clone(),
    };

    println!("change_txout: {:?}", change_txout);

    let mut tx = Transaction {
        version: transaction::Version::ONE,
        lock_time: absolute::LockTime::Blocks(Height::from_consensus(1).unwrap()),
        input: vec![txin],
        output: vec![txout, change_txout],
    };
    println!("tx: {:?}", tx.clone());

    // Get the sighash to sign.
    let sighash_type = EcdsaSighashType::All;
    let sighasher = SighashCache::new(&mut tx);
    let sighash = sighasher
        .legacy_signature_hash(0, &bob_script_pubkey, sighash_type.to_u32())
        .expect("failed to create sighash");

    println!("sighash: {:?}", sighash);

    let msg = Message::from(sighash);
    println!("msg: {:?}", msg);

    let signature = secp.sign_ecdsa(&msg, &first_priv_key);
    println!("signature: {:?}", signature);

    // Verify signature
    let is_valid = secp.verify_ecdsa(&msg, &signature, &public_key).is_ok();
    println!("Signature valid: {:?}", is_valid);

    assert!(is_valid, "The signature should be valid");

    let signature = bitcoin::ecdsa::Signature {
        signature,
        sighash_type,
    };

    // Create the script_sig
    let script_sig = Builder::new()
        .push_slice(&signature.serialize())
        .push_key(&bitcoin_public_key)
        .into_script();

    // Verify the script_sig
    println!("script_sig: {:?}", script_sig);

    // Decode the script_sig to verify its contents
    let mut iter = script_sig.instructions().peekable();

    // Check the signature
    if let Some(Ok(Instruction::PushBytes(sig_bytes))) = iter.next() {
        println!("Signature in script_sig: {:?}", sig_bytes);

        assert_eq!(
            sig_bytes.as_bytes(),
            signature.serialize().to_vec().as_slice(),
            "Signature mismatch in script_sig"
        );
    } else {
        panic!("Expected signature push in script_sig");
    }

    // Check the public key
    if let Some(Ok(Instruction::PushBytes(pubkey_bytes))) = iter.next() {
        println!("Public key in script_sig: {:?}", pubkey_bytes);
        assert_eq!(
            pubkey_bytes.as_bytes(),
            bitcoin_public_key.to_bytes(),
            "Public key mismatch in script_sig"
        );
    } else {
        panic!("Expected public key push in script_sig");
    }

    // Ensure there are no more instructions
    assert!(iter.next().is_none(), "Unexpected data in script_sig");

    println!("script_sig verification passed");

    // Assign script_sig to txin
    tx.input[0].script_sig = script_sig;

    // Finalize the transaction
    let tx_signed = tx;

    println!("tx_signed: {:?}", tx_signed);

    let raw_tx_result = client.send_raw_transaction(&tx_signed).unwrap();
    println!("raw_tx_result: {:?}", raw_tx_result);

    client.generate_to_address(101, &bob_address)?;

    assert_utxos_for_address(client, alice_address, 1);

    Ok(())
}

fn assert_utxos_for_address(client: &bitcoind::Client, address: Address, number_of_utxos: usize) {
    let min_conf = 1;
    let max_conf = 9999999;
    let include_unsafe = true;
    let query_options = json!({});

    let unspent_utxos: Vec<serde_json::Value> = client
        .call(
            "listunspent",
            &[
                json!(min_conf),
                json!(max_conf),
                json!(vec![address.to_string()]),
                json!(include_unsafe),
                query_options.clone(),
            ],
        )
        .unwrap();

    assert!(
        unspent_utxos.len() == number_of_utxos,
        "Expected {} UTXOs for address {}, but found {}",
        number_of_utxos,
        address.to_string(),
        unspent_utxos.len()
    );
}
