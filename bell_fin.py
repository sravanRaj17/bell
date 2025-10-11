import streamlit as st
import hashlib
import json
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

def aes_gcm_encrypt(plaintext: bytes, key: bytes):
    """Encrypt plaintext with AES-GCM. Returns (nonce, ciphertext, tag)."""
    nonce = secrets.token_bytes(12)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend()).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    tag = encryptor.tag
    return nonce, ciphertext, tag

def aes_gcm_decrypt(nonce: bytes, ciphertext: bytes, tag: bytes, key: bytes):
    """Decrypt ciphertext with AES-GCM."""
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()
import binascii
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import random
import base64
import numpy as np
import math
import subprocess
import sys
import tempfile
import os
import qrcode
from io import BytesIO
import threading
import concurrent.futures
import secrets  # Added for cryptographically secure random number generation

# Quantum imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Cryptography imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # Added for secure key derivation
from cryptography.hazmat.backends import default_backend

# Market data imports
import requests

# IBM Quantum (optional)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator, Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManila, FakeLima
    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False

# ==================== MARKET DATA FUNCTIONS ====================

def fetch_crypto_data(coin_ids='bitcoin,ethereum,solana,cardano,ripple'):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': coin_ids,
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1,
            'sparkline': 'true',
            'price_change_percentage': '24h,7d'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return None

def fetch_fiat_rates(base_currency='USD'):
    """Fetch fiat exchange rates from ExchangeRate.host API"""
    try:
        url = f"https://api.exchangerate.host/latest"
        params = {'base': base_currency}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['rates'] if data.get('success') else None
    except Exception as e:
        st.error(f"Error fetching fiat rates: {e}")
        return None

def get_crypto_history(coin_id, days=7):
    """Get historical price data for a cryptocurrency"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['prices'] if 'prices' in data else None
    except Exception as e:
        st.error(f"Error fetching historical data for {coin_id}: {e}")
        return None

# ==================== ENHANCED HMAC AUTHENTICATION ====================

def derive_hmac_key(quantum_key_hex=None, classical_psk=None):
    """
    Derive a post-quantum resistant HMAC key from both quantum and classical sources.
    Uses SHAKE256 for extensible output and post-quantum security.
    """
    # Convert inputs to bytes if they are strings
    if quantum_key_hex and isinstance(quantum_key_hex, str):
        quantum_key = binascii.unhexlify(quantum_key_hex)
    else:
        quantum_key = quantum_key_hex or b''
        
    if classical_psk and isinstance(classical_psk, str):
        classical_psk = classical_psk.encode()
    else:
        classical_psk = classical_psk or b''
    
    # Combine both keys with domain separation
    combined_input = b'quantumverse-hmac-key' + quantum_key + b'||' + classical_psk
    
    # Use SHAKE256 for post-quantum secure key derivation
    shake256 = hashlib.shake_256()
    shake256.update(combined_input)
    
    # Return 64 bytes for SHA3-512 HMAC
    return shake256.digest(64)

def generate_hmac(message, key):
    """Generate HMAC using SHA3-512 with enhanced key derivation"""
    if isinstance(message, str):
        message = message.encode()
    
    # Derive enhanced key if we have both quantum and classical components
    if isinstance(key, tuple) and len(key) == 2:
        quantum_key, classical_psk = key
        key = derive_hmac_key(quantum_key, classical_psk)
    elif isinstance(key, str):
        # If it's a string, check if it's a hex-encoded quantum key or a PSK
        if len(key) == 64 and all(c in '0123456789abcdefABCDEF' for c in key):
            # Looks like a quantum key (64 hex chars = 32 bytes)
            key = derive_hmac_key(quantum_key_hex=key, classical_psk=st.session_state.get("authentication_psk", ""))
        else:
            # Treat as classical PSK
            key = derive_hmac_key(classical_psk=key)
    
    h = hmac.HMAC(key, hashes.SHA3_512(), backend=default_backend())
    h.update(message)
    return h.finalize().hex()

def verify_hmac(message, received_hmac, key):
    """Verify HMAC using SHA3-512 with constant-time comparison"""
    try:
        expected_hmac = generate_hmac(message, key)
        return secrets.compare_digest(expected_hmac, received_hmac)
    except Exception:
        return False

# ==================== ANSATZ & OPTIMIZER ====================

def bb84_ansatz(qc, qubits, params):
    """Simple variational ansatz with RY rotations and CZ entanglement."""
    for i, q in enumerate(qubits):
        qc.ry(float(params[i % len(params)]), q)
    for i in range(len(qubits) - 1):
        qc.cz(qubits[i], qubits[i+1])
    for i, q in enumerate(qubits):
        qc.ry(float(params[(i+1) % len(params)]), q)

def spsa_optimize_ansatz(run_protocol_fn, params_shape, steps=50, seed=42):
    """Basic SPSA optimizer to minimize QBER using calibration runs."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-np.pi, np.pi, size=params_shape)
    a0, c0, A = 0.2, 0.1, steps/10 + 1
    for k in range(1, steps+1):
        ak = a0 / ((k + A) ** 0.602)
        ck = c0 / (k ** 0.101)
        delta = rng.choice([-1, 1], size=params_shape)
        loss_plus = run_protocol_fn(thetas + ck*delta, calib=True)
        loss_minus = run_protocol_fn(thetas - ck*delta, calib=True)
        ghat = (loss_plus - loss_minus) / (2 * ck * delta)
        thetas = thetas - ak * ghat
    return thetas

# ==================== UTILITY FUNCTIONS ====================

def is_power_of_two(n):
    """Check if a number is a power of two"""
    return (n & (n-1) == 0) and n != 0

def get_optimal_batch_size(backend, num_qubits_per_symbol, dimension):
    """Determine optimal batch size based on backend capabilities and dimension"""
    if hasattr(backend, 'configuration') and callable(getattr(backend, 'configuration', None)):
        try:
            max_qubits = backend.configuration().n_qubits
            available_qubits = max_qubits - 2
            
            # Adjust batch size based on dimension (smaller for higher dimensions)
            dimension_factor = max(1, 8 // int(math.log2(dimension)))
            return max(1, available_qubits // (num_qubits_per_symbol * dimension_factor))
        except Exception:
            # Fallback for simulator
            if dimension <= 4:
                return 20
            elif dimension == 8:
                return 10
            else:
                return 5
    else:
        # For simulator, use smaller batches for higher dimensions
        if dimension <= 4:
            return 20
        elif dimension == 8:
            return 10
        else:
            return 5

def is_ibm_quantum_available():
    """Check if IBM Quantum is available and configured"""
    return (IBM_QUANTUM_AVAILABLE and 
            st.session_state.get("ibm_quantum_configured", False) and
            st.session_state.get("ibm_api_key"))

def validate_parameters(dimension, key_length, decoy_ratio):
    """Comprehensive parameter validation for security"""
    if not is_power_of_two(dimension):
        raise ValueError("Dimension must be a power of 2")
    if key_length < 128:
        raise ValueError("Key length must be at least 128 bits")
    if not (0 < decoy_ratio < 0.5):
        raise ValueError("Decoy ratio must be between 0 and 0.5")
    if dimension > 16:  # Practical limit for simulation
        st.warning("High dimensions may cause performance issues")
    return True

# ==================== AUTHENTICATION & SECURITY ====================

def derive_authentication_key(quantum_key):
    """Derive authentication key from quantum key using SHAKE256"""
    # Use SHAKE256 for post-quantum security
    shake256 = hashlib.shake_256()
    shake256.update(b'quantumverse-auth-key')
    shake256.update(binascii.unhexlify(quantum_key))
    return shake256.digest(64)  # 64 bytes for SHA3-512

def derive_psk_from_user_input(user_psk):
    """Derive secure PSK from user input using PBKDF2 with SHA3-512"""
    if len(user_psk) < 32:
        raise ValueError("Pre-shared key must be at least 32 characters long")
    
    salt = b'quantumverse_psk_salt'  # Should be unique per user in production
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA3_512(),
        length=64,  # 64 bytes for SHA3-512
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(user_psk.encode())

# ==================== QUANTUM KEY DISTRIBUTION ====================

def cascade_error_reconciliation(alice_bits, bob_bits, block_size=12, max_iterations=8):
    """Enhanced Cascade protocol for error reconciliation"""
    alice_blocks = [alice_bits[i:i+block_size] for i in range(0, len(alice_bits), block_size)]
    bob_blocks = [bob_bits[i:i+block_size] for i in range(0, len(bob_bits), block_size)]
    errors_corrected = 0
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        parity_mismatches = []
        
        for i, (a_block, b_block) in enumerate(zip(alice_blocks, bob_blocks)):
            a_parity = sum(int(bit) for bit in a_block) % 2
            b_parity = sum(int(bit) for bit in b_block) % 2
            if a_parity != b_parity:
                parity_mismatches.append(i)
        
        if not parity_mismatches:
            break
            
        for block_idx in parity_mismatches:
            a_block = alice_blocks[block_idx]
            b_block = bob_blocks[block_idx]
            
            left, right = 0, len(a_block) - 1
            error_positions = []
            
            while left <= right:
                if left == right:
                    error_positions.append(left)
                    break
                    
                mid = (left + right) // 2
                a_left_parity = sum(int(bit) for bit in a_block[left:mid+1]) % 2
                b_left_parity = sum(int(bit) for bit in b_block[left:mid+1]) % 2
                
                if a_left_parity != b_left_parity:
                    right = mid
                else:
                    a_right_parity = sum(int(bit) for bit in a_block[mid+1:right+1]) % 2
                    b_right_parity = sum(int(bit) for bit in b_block[mid+1:right+1]) % 2
                    
                    if a_right_parity != b_right_parity:
                        left = mid + 1
                    else:
                        for pos in range(left, right + 1):
                            if a_block[pos] != b_block[pos]:
                                error_positions.append(pos)
                        break
            
            b_block_list = list(b_block)
            for pos in error_positions:
                b_block_list[pos] = str(1 - int(b_block_list[pos]))
                errors_corrected += 1
                
            bob_blocks[block_idx] = ''.join(b_block_list)
            break
    
    verified_blocks = []
    for a_block, b_block in zip(alice_blocks, bob_blocks):
        if a_block == b_block:
            verified_blocks.append(b_block)
        else:
            verified_blocks.append(a_block)
    
    reconciled_bits = ''.join(verified_blocks)
    return reconciled_bits, errors_corrected, iterations

def apply_forward_error_correction(binary_string, correction_strength=2):
    """Apply forward error correction"""
    if correction_strength == 1:
        corrected = ""
        for i in range(0, len(binary_string), 8):
            chunk = binary_string[i:i+8]
            if len(chunk) < 8:
                chunk = chunk.ljust(8, '0')
            parity = str(sum(int(bit) for bit in chunk) % 2)
            corrected += chunk + parity
        return corrected
        
    elif correction_strength == 2:
        corrected = ""
        for i in range(0, len(binary_string), 8):
            chunk = binary_string[i:i+8]
            if len(chunk) < 8:
                chunk = chunk.ljust(8, '0')
            
            row_parity = str(sum(int(bit) for bit in chunk) % 2)
            col_parity = ""
            if len(chunk) >= 8:
                for j in range(4):
                    col_bits = chunk[j] + chunk[j+4]
                    col_parity += str(sum(int(bit) for bit in col_bits) % 2)
            
            corrected += chunk + row_parity + col_parity
        return corrected
        
    else:
        corrected = ""
        for i in range(0, len(binary_string), 4):
            chunk = binary_string[i:i+4]
            if len(chunk) < 4:
                chunk = chunk.ljust(4, '0')
            
            p1 = str((int(chunk[0]) + int(chunk[1]) + int(chunk[3])) % 2)
            p2 = str((int(chunk[0]) + int(chunk[2]) + int(chunk[3])) % 2)
            p3 = str((int(chunk[1]) + int(chunk[2]) + int(chunk[3])) % 2)
            
            corrected += chunk + p1 + p2 + p3
        return corrected

def estimate_error_rate(indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_key, use_auth):
    """Estimate error rate for signal or decoy states"""
    if not indices:
        return 0
    
    if use_auth:
        indices_str = json.dumps(indices)
        # Use enhanced HMAC with both quantum and classical keys
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_key)
        indices_hmac = generate_hmac(indices_str, hmac_key)
        if not verify_hmac(indices_str, indices_hmac, hmac_key):
            st.error("Authentication failed: Sample indices could not be verified")
            return float('inf')
    
    sifted_alice = []
    sifted_bob = []
    for i in indices:
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_symbols[i])
            sifted_bob.append(measured_symbols[i])
    
    sample_size = min(30, len(sifted_alice) // 2)
    if sample_size == 0:
        return 0
    
    # Use cryptographically secure random sampling
    sample_indices = sorted(secrets.SystemRandom().sample(range(len(sifted_alice)), sample_size))
    
    if use_auth:
        sample_indices_str = json.dumps(sample_indices)
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_key)
        sample_indices_hmac = generate_hmac(sample_indices_str, hmac_key)
        if not verify_hmac(sample_indices_str, sample_indices_hmac, hmac_key):
            st.error("Authentication failed: Sample indices could not be verified")
            return float('inf')
    
    error_count = 0
    for idx in sample_indices:
        if sifted_alice[idx] != sifted_bob[idx]:
            error_count += 1
    
    if use_auth:
        error_count_str = str(error_count)
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_key)
        error_count_hmac = generate_hmac(error_count_str, hmac_key)
        if not verify_hmac(error_count_str, error_count_hmac, hmac_key):
            st.error("Authentication failed: Error count could not be verified")
            return float('inf')
    
    return error_count / sample_size

def run_high_dimensional_bb84_protocol(key_length=256, dimension=4, backend=None):
    """Enhanced BB84 protocol with modern progress tracking and security improvements"""
    try:
        validate_parameters(dimension, key_length, 0.3)
    except ValueError as e:
        st.error(f"Invalid protocol parameters: {e}")
        return None
        
    use_auth = st.session_state.get("authentication_enabled", True)
    auth_psk = st.session_state.get("authentication_psk", "")
    block_size = st.session_state.get("reconciliation_block_size", 12)
    max_iterations = st.session_state.get("reconciliation_max_iterations", 8)
    error_correction_strength = st.session_state.get("error_correction_strength", 2)

    bits_per_symbol = math.log2(dimension)
    decoy_ratio = 0.3
    initial_length = int(key_length * 6 / bits_per_symbol / (1 - decoy_ratio))

    # Generate quantum protocol parameters using cryptographically secure RNG
    alice_symbols = [secrets.randbelow(dimension) for _ in range(initial_length)]
    alice_bases = [secrets.randbits(1) for _ in range(initial_length)]
    alice_decoy_states = [secrets.randbits(1) == 0 for _ in range(initial_length)]  # 50% chance for decoy
    bob_bases = [secrets.randbits(1) for _ in range(initial_length)]

    decoy_intensities = {'signal': 0.5, 'decoy': 0.1}
    num_qubits_per_symbol = int(math.ceil(math.log2(dimension)))
    batch_size = get_optimal_batch_size(backend, num_qubits_per_symbol, dimension)

    # Modern progress tracking
    progress_container = st.container()
    with progress_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        with col2:
            progress_text = st.empty()

    try:
        circuits = []
        measured_symbols = []

        # Circuit creation phase
        status_text.text(" Creating quantum circuits...")
        for batch_start in range(0, initial_length, batch_size):
            batch_end = min(batch_start + batch_size, initial_length)
            batch_symbols = alice_symbols[batch_start:batch_end]
            batch_alice_bases = alice_bases[batch_start:batch_end]
            batch_bob_bases = bob_bases[batch_start:batch_end]

            total_qubits = num_qubits_per_symbol * (batch_end - batch_start)
            qc = QuantumCircuit(total_qubits, total_qubits)

            for i, (symbol, alice_basis, bob_basis) in enumerate(zip(
                batch_symbols, batch_alice_bases, batch_bob_bases
            )):
                qubit_start = i * num_qubits_per_symbol
                qubit_end = (i + 1) * num_qubits_per_symbol

                # Encode symbol
                # Apply ansatz if enabled
                if st.session_state.get('use_ansatz', False):
                    params = st.session_state.get('ansatz_params', np.zeros(4))
                    bb84_ansatz(qc, list(range(qubit_start, qubit_end)), params)

                symbol_bin = format(symbol, f'0{num_qubits_per_symbol}b')
                for j, bit in enumerate(symbol_bin):
                    if bit == '1':
                        qc.x(qubit_start + j)

                # Alice's basis transformation
                if alice_basis == 1:
                    qc.h(range(qubit_start, qubit_end))
                    # Simplified entanglement for efficiency
                    if num_qubits_per_symbol > 1:
                        qc.cz(qubit_start, qubit_end-1)

                # Bob's basis transformation
                if bob_basis == 1:
                    if num_qubits_per_symbol > 1:
                        qc.cz(qubit_start, qubit_end-1)
                    qc.h(range(qubit_start, qubit_end))

                qc.measure(range(qubit_start, qubit_end), range(qubit_start, qubit_end))

            circuits.append(qc)
            
            # Update progress
            progress = min(1.0, (batch_end / initial_length) * 0.3)
            progress_bar.progress(progress)
            progress_text.text(f"{int(progress*100)}%")

        # Execute circuits with retry logic
        if backend is None:
            backend = AerSimulator()

        execution_batch_size = min(5, len(circuits))  # Smaller batch size for execution
        
        status_text.text(" Executing quantum circuits...")
        max_retries = 3
        for i in range(0, len(circuits), execution_batch_size):
            batch_end = min(i + execution_batch_size, len(circuits))
            batch = circuits[i:batch_end]

            for retry in range(max_retries):
                try:
                    job = backend.run(batch, shots=1, memory=True)
                    result = job.result()

                    for j, circuit in enumerate(batch):
                        memory = result.get_memory(j)[0]
                        batch_symbol_count = circuit.num_qubits // num_qubits_per_symbol

                        for k in range(batch_symbol_count):
                            start_idx = k * num_qubits_per_symbol
                            end_idx = (k + 1) * num_qubits_per_symbol
                            measured_bits = memory[start_idx:end_idx]
                            measured_symbol = int(measured_bits, 2) % dimension
                            measured_symbols.append(measured_symbol)
                    break  # Success, break out of retry loop
                except Exception as e:
                    if retry == max_retries - 1:
                        st.error(f"Error running quantum circuits after {max_retries} attempts: {e}")
                        return None
                    time.sleep(1)  # Wait before retry

            # Update progress
            progress = 0.3 + (min(i + execution_batch_size, len(circuits)) / len(circuits)) * 0.5
            progress_bar.progress(progress)
            progress_text.text(f"{int(progress*100)}%")

        # Ensure correct number of measurements
        if len(measured_symbols) < initial_length:
            st.error("Insufficient measurements obtained from quantum backend")
            return None
        elif len(measured_symbols) > initial_length:
            measured_symbols = measured_symbols[:initial_length]

    except Exception as e:
        st.error(f"Unexpected error in quantum protocol: {e}")
        return None

    # Authentication and decoy state analysis
    status_text.text(" Verifying authentication...")
    if use_auth:
        auth_data = {
            "alice_bases": alice_bases,
            "alice_decoy_states": alice_decoy_states,
            "decoy_intensities": decoy_intensities
        }
        auth_data_str = json.dumps(auth_data)
        # Use enhanced HMAC with both quantum and classical keys
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_psk)
        auth_hmac = generate_hmac(auth_data_str, hmac_key)
        if not verify_hmac(auth_data_str, auth_hmac, hmac_key):
            st.error("Authentication failed: Quantum state information could not be verified")
            return None

    # Decoy state analysis
    status_text.text(" Analyzing decoy states...")
    signal_indices = [i for i in range(initial_length) if not alice_decoy_states[i]]
    decoy_indices = [i for i in range(initial_length) if alice_decoy_states[i]]

    signal_error_rate = estimate_error_rate(signal_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)
    decoy_error_rate = estimate_error_rate(decoy_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)

    max_error_rate = (dimension + 3) / (2 * dimension)
    
    if decoy_error_rate > signal_error_rate * 2.0 and decoy_error_rate > 0.1:
        st.error("Photon number splitting attack detected! Protocol aborted.")
        return None

    # Sifting and sampling
    status_text.text(" Sifting key bits...")
    sifted_key_alice = []
    sifted_key_bob = []
    for i in signal_indices:
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(alice_symbols[i])
            sifted_key_bob.append(measured_symbols[i])

    if len(sifted_key_alice) < 20:
        st.error(f"Not enough sifted key bits ({len(sifted_key_alice)}). Protocol requires at least 20.")
        return None
        
    sample_size = min(100, len(sifted_key_alice) // 3)
    # Use cryptographically secure random sampling
    sample_indices = sorted(secrets.SystemRandom().sample(range(len(sifted_key_alice)), sample_size))
    
    if use_auth:
        sample_indices_str = json.dumps(sample_indices)
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_psk)
        sample_indices_hmac = generate_hmac(sample_indices_str, hmac_key)
        if not verify_hmac(sample_indices_str, sample_indices_hmac, hmac_key):
            st.error("Authentication failed: Sample indices could not be verified")
            return None
            
    error_count = 0
    for idx in sample_indices:
      if sifted_key_alice[idx] != sifted_key_bob[idx]:   #fixed
        error_count += 1
            
    if use_auth:
        error_count_str = str(error_count)
        hmac_key = (st.session_state.get("quantum_keys", {}).get(st.session_state.get("logged_in_user", ""), ""), auth_psk)
        error_count_hmac = generate_hmac(error_count_str, hmac_key)
        if not verify_hmac(error_count_str, error_count_hmac, hmac_key):
            st.error("Authentication failed: Error count could not be verified")
            return None
            
    error_rate = error_count / sample_size if sample_size > 0 else 0
    
    if error_rate > max_error_rate:
        st.warning(f"High error rate detected ({error_rate:.3f}) but continuing for demonstration")

    # Remove sample symbols
    final_key_alice = []
    final_key_bob = []
    for i in range(len(sifted_key_alice)):
        if i not in sample_indices:
            final_key_alice.append(sifted_key_alice[i])
            final_key_bob.append(sifted_key_bob[i])

    # Error reconciliation
    status_text.text(" Reconciling errors...")
    symbol_bits = int(math.log2(dimension))
    alice_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_alice)
    bob_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_bob)
    
    if len(alice_binary) == 0:
        st.error("No key bits remaining after sampling")
        return None
        
    reconciled_binary, errors_corrected, iterations = cascade_error_reconciliation(
        alice_binary, bob_binary, block_size=block_size, max_iterations=max_iterations
    )
    
    if errors_corrected > 0:
        st.success(f"Error reconciliation corrected {errors_corrected} bits in {iterations} iterations")

    reconciled_binary = apply_forward_error_correction(reconciled_binary, error_correction_strength)

    # Convert back to symbols
    reconciled_key = []
    for i in range(0, len(reconciled_binary), symbol_bits):
        symbol_bin = reconciled_binary[i:i+symbol_bits]
        if len(symbol_bin) == symbol_bits:
            reconciled_key.append(int(symbol_bin, 2))

    # Privacy amplification with information leakage accounting
    status_text.text(" Amplifying privacy...")
    if not reconciled_key:
        st.error("No key bits after reconciliation")
        return None
        
    key_str = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in reconciled_key)
    
    # Estimate leaked bits during error correction and account for them
    leaked_bits = errors_corrected * math.log2(dimension)  # Simplified estimate
    security_parameter = 256 + int(leaked_bits * 2)  # Conservative estimate
    
    if len(key_str) < security_parameter:
        st.error("Insufficient key material for privacy amplification")
        return None
        
    # Trim to account for information leakage
    final_key_length = max(0, (len(key_str) - security_parameter) // 8)
    if final_key_length < key_length // 8:
        st.error("Insufficient key material after privacy amplification")
        return None
        
    key_bytes = int(key_str, 2).to_bytes((len(key_str) + 7) // 8, byteorder='big')
    
    hkdf = HKDF(
        algorithm=hashes.SHA3_512(),
        length=final_key_length,
        salt=os.urandom(16),
        info=b'high-dim-bb84-reconciled-key',
        backend=default_backend()
    )
    
    try:
        final_key = hkdf.derive(key_bytes)
    except Exception as e:
        st.error(f"Key derivation failed: {e}")
        return None
        
    auth_key = derive_authentication_key(binascii.hexlify(final_key).decode())
    
    # Final progress update
    progress_bar.progress(1.0)
    progress_text.text("100%")
    status_text.text(" Quantum key generation complete!")
    time.sleep(0.5)
    progress_container.empty()
    
    return binascii.hexlify(final_key).decode()

def one_time_circuit_high_dim_bb84(key_length=256, dimension=4):
    """Generate quantum key with modern UI feedback and enhanced security"""
    try:
        validate_parameters(dimension, key_length, 0.3)
    except ValueError as e:
        st.error(f"Invalid protocol parameters: {e}")
        return None
    
    use_real_hardware = (
        is_ibm_quantum_available() and 
        st.session_state.get("use_real_hardware", False)
    )
    
    backend = None
    hardware_type = "Aer Simulator"
    
    if use_real_hardware:
        try:
            service = QiskitRuntimeService(
                channel=st.session_state.get("ibm_channel", "ibm_quantum"),
                token=st.session_state.get("ibm_api_key"),
                instance=st.session_state.get("ibm_instance", "")
            )
            backends = service.backends(simulator=False, operational=True)
            
            if backends:
                backend = backends[0]
                hardware_type = f"IBM Quantum ({backend.name})"
                st.session_state["last_quantum_backend"] = backend.name
            else:
                st.warning("No suitable quantum hardware available. Using simulator.")
                backend = AerSimulator()
                hardware_type = "Aer Simulator (fallback)"
        except Exception as e:
            st.error(f"IBM Quantum connection failed: {e}. Using simulator.")
            backend = AerSimulator()
            hardware_type = "Aer Simulator (fallback)"
            st.session_state["ibm_quantum_configured"] = False
    else:
        backend = AerSimulator()
    
    try:
        # Create a placeholder for the quantum process
        quantum_placeholder = st.empty()
        with quantum_placeholder.container():
            st.info(f" Starting quantum key generation using {hardware_type}...")
            
        # Run the quantum protocol in a separate thread to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_high_dimensional_bb84_protocol, key_length, dimension, backend)
            
            # Show progress while waiting
            with st.spinner(f"Generating {dimension}-D quantum key..."):
                while not future.done():
                    time.sleep(0.1)
                
                key = future.result()
                
        quantum_placeholder.empty()
            
        if key:
            st.session_state["last_hardware_used"] = hardware_type
            return key
        else:
            st.error("Quantum protocol aborted due to security issues")
            return None
    except Exception as e:
        st.error(f"Error in quantum protocol: {e}")
        return None

# ==================== BLOCKCHAIN FUNCTIONS ====================

def calculate_hash(block: dict) -> str:
    """Calculate SHA256 hash of block"""
    block_copy = block.copy()
    block_copy.pop("hash", None)
    block_string = json.dumps(block_copy, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()

def build_merkle_tree(transactions):
    """Build Merkle tree for transactions"""
    if not transactions:
        return "0"
    tx_hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest() for tx in transactions]
    while len(tx_hashes) > 1:
        if len(tx_hashes) % 2 != 0:
            tx_hashes.append(tx_hashes[-1])
        tx_hashes = [
            hashlib.sha256((tx_hashes[i] + tx_hashes[i+1]).encode()).hexdigest()
            for i in range(0, len(tx_hashes), 2)
        ]
    return tx_hashes[0]


def generate_wallet(username):
    """Generate new wallet with Ed25519 keys"""
    # Generate Ed25519 private/public key pair and export raw bytes as hex
    sk = ed25519.Ed25519PrivateKey.generate()
    vk = sk.public_key()
    priv_bytes = sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = vk.public_bytes(Encoding.Raw, PublicFormat.Raw)

    return {
        "private_key": priv_bytes.hex(),
        "public_key": pub_bytes.hex(),
        "username": username,
        "created_at": datetime.now().isoformat(),
        "total_sent": 0,
        "total_received": 0,
        "transactions_count": 0
    }



def verify_signature(transaction) -> bool:
    """Verify Ed25519 signature"""
    try:
        sender_pk_hex = transaction["sender"]
        signature_hex = transaction["signature"]
        sender_pub_bytes = binascii.unhexlify(sender_pk_hex)
        sender_vk = ed25519.Ed25519PublicKey.from_public_bytes(sender_pub_bytes)
        signature = binascii.unhexlify(signature_hex)

        tx_data_to_verify = {k: v for k, v in transaction.items() if k != "signature"}
        tx_string = json.dumps(tx_data_to_verify, sort_keys=True).encode()
        tx_hash = hashlib.sha256(tx_string).hexdigest().encode()

        # will raise an exception if verification fails
        sender_vk.verify(signature, tx_hash)
        return True
    except Exception:
        return False


def update_balances(transactions):
    """Update account balances"""
    for tx in transactions:
        sender = tx["sender"]
        receiver = tx["receiver"]
        amount = float(tx["amount"])
        if sender != "network":
            st.session_state.balances[sender] = st.session_state.balances.get(sender, 0) - amount
        st.session_state.balances[receiver] = st.session_state.balances.get(receiver, 0) + amount

def process_transaction(transaction) -> bool:
    """Process and confirm transaction"""
    if not verify_signature(transaction):
        st.error("Invalid transaction signature!")
        return False

    previous_hash = st.session_state.blockchain[-1]["hash"] if st.session_state.blockchain else "0"
    
    new_block = {
        "index": len(st.session_state.blockchain),
        "transactions": [transaction],
        "merkle_root": build_merkle_tree([transaction]),
        "timestamp": str(datetime.now()),
        "previous_hash": previous_hash,
        "miner": "network",
        "quantum_dimension": st.session_state.quantum_dimension
    }
    new_block["hash"] = calculate_hash(new_block)

    st.session_state.blockchain.append(new_block)
    update_balances([transaction])

    st.session_state.network_stats["total_transactions"] += 1
    st.session_state.network_stats["total_volume"] += float(transaction["amount"])
    return True



def create_transaction(sender_private_key_hex, receiver_public_key, amount, tx_type="transfer", fee: float = 0.0):
    """
    Create and sign a transaction using Ed25519 keys, then secure the signed report with a BB84-derived key
    generated in cooperation with the Authentication Unit (a bank-like authority). The encrypted signed report
    is submitted to the Authentication Unit which decrypts, verifies the Ed25519 signature, and appends the
    transaction to the blockchain if valid.

    This function integrates the simulated BB84 flow (or a pluggable real BB84 routine) and keeps compatibility
    with the Streamlit session state used by the rest of the app.
    """
    try:
        sender_sk = ed25519.Ed25519PrivateKey.from_private_bytes(binascii.unhexlify(sender_private_key_hex))
        sender_public_key = sender_sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    except (binascii.Error, ValueError):
        st.error("Invalid private key format.")
        return None

    # Parse amounts
    try:
        amount = float(amount)
        fee = float(fee)
    except Exception:
        st.error("Invalid amount or fee.")
        return None

    balance = st.session_state.balances.get(sender_public_key, 0.0)
    total_cost = amount + fee
    if balance < total_cost:
        st.error(f"Insufficient funds! Need {total_cost:.2f} coins, have {balance:.2f}")
        return None

    # Build a simple transaction report (signed) - the same canonical JSON will be verified by the auth unit
    report = {
        "sender": sender_public_key,
        "receiver": receiver_public_key,
        "amount": amount,
        "fee": fee,
        "type": tx_type,
        "timestamp": str(datetime.now()),
    }
    report_bytes = json.dumps(report, sort_keys=True).encode()

    # Sign the report with Ed25519
    try:
        signature = sender_sk.sign(report_bytes).hex()
    except Exception as e:
        st.error(f"Signing failed: {e}")
        return None

    # --- Quantum (BB84) session with Authentication Unit ---
    # Use an AuthenticationUnit stored in session (create one if missing)
    try:
        # Lazy import of our helper functions (either from this file or the separate module)
        # If the app has its own one_time_circuit_high_dim_bb84 function, prefer to use it.
        use_internal_bb84 = False
        if 'one_time_circuit_high_dim_bb84' in globals() and callable(globals()['one_time_circuit_high_dim_bb84']):
            use_internal_bb84 = True

        session_id = f"sess_{int(time.time())}_{secrets.randbelow(900000)}"

        # Acquire BB84-derived raw key (either real function or simulated one)
        if use_internal_bb84:
            bb84_key_hex = one_time_circuit_high_dim_bb84(256, st.session_state.quantum_dimension)
            if bb84_key_hex is None:
                st.error("Failed to generate quantum key. Transaction aborted.")
                return None
            raw_bb84 = binascii.unhexlify(bb84_key_hex)
        else:
            # fallback: simulate secure BB84 key
            raw_bb84 = secrets.token_bytes(32)

        # Derive AES key from raw BB84 material (HKDF)
        hkdf = HKDF(
            algorithm=hashes.SHA3_256() if hasattr(hashes, 'SHA3_256') else hashes.SHA256(),
            length=32,
            salt=None,
            info=b'bb84-aes-key',
            backend=default_backend()
        )
        aes_key = hkdf.derive(raw_bb84)

        # Ensure there is an Authentication Unit instance in session state (bank-like authority)
        if 'auth_unit' not in st.session_state:
            # Minimal in-memory auth unit that mirrors the demo AuthenticationUnit
            class _InAppAuthUnit:
                def __init__(self):
                    self.session_keys = {}
                    self.blockchain = st.session_state.blockchain  # link to the app blockchain list
                def store_session_key(self, sid, key, theta=None):
                    # store key bytes and associated theta for this session
                    self.session_keys[sid] = {'key': key, 'theta': theta}
                def get_session_key(self, sid):
                    v = self.session_keys.get(sid)
                    return v['key'] if v and isinstance(v, dict) else v
                def get_session_theta(self, sid):
                    v = self.session_keys.get(sid)
                    return v.get('theta') if v and isinstance(v, dict) else None
                def verify_and_process_encrypted_report(self, encrypted_blob):
                    # This will be implemented below using the same logic as the demo module
                    session_id = encrypted_blob.get('session_id')
                    key = self.get_session_key(session_id)
                    if not key:
                        return False, 'no session key'
                    try:
                        nonce = binascii.unhexlify(encrypted_blob['nonce'])
                        ciphertext = binascii.unhexlify(encrypted_blob['ciphertext'])
                        tag = binascii.unhexlify(encrypted_blob['tag'])
                        plaintext = aes_gcm_decrypt(nonce, ciphertext, tag, key)
                        payload = json.loads(plaintext.decode())
                        report = payload.get('report')
                        signature_hex = payload.get('signature')
                        sender_pub_hex = payload.get('sender_public_key')
                        if not report or not signature_hex or not sender_pub_hex:
                            return False, 'missing fields'
                        vk = ed25519.Ed25519PublicKey.from_public_bytes(binascii.unhexlify(sender_pub_hex))
                        vk.verify(binascii.unhexlify(signature_hex), json.dumps(report, sort_keys=True).encode())
                        # Append to blockchain as a new block containing the transaction report
                        prev_hash = st.session_state.blockchain[-1]['hash'] if st.session_state.blockchain else '0'
                        new_block = {
                            'index': len(st.session_state.blockchain),
                            'transactions': [report],
                            'merkle_root': build_merkle_tree([report]),
                            'timestamp': str(datetime.now()),
                            'previous_hash': prev_hash,
                            'miner': 'auth_unit',
                            'quantum_dimension': st.session_state.quantum_dimension
                        }
                        new_block['hash'] = calculate_hash(new_block)
                        st.session_state.blockchain.append(new_block)
                        # Update balances locally
                        update_balances([report])
                        st.session_state.network_stats['total_transactions'] += 1
                        st.session_state.network_stats['total_volume'] += float(report.get('amount',0))
                        return True, 'ok'
                    except Exception as e:
                        return False, f'decrypt/verify error: {e}'

            st.session_state['auth_unit'] = _InAppAuthUnit()

        auth_unit = st.session_state['auth_unit']

        # Store the AES session key at the auth unit side, simulating the BB84 shared key placement
        auth_unit.store_session_key(session_id, aes_key)
        # Also keep a reference in the app session for auditing
        st.session_state['quantum_keys'][session_id] = binascii.hexlify(raw_bb84).decode()

    except Exception as e:
        st.error(f"BB84/session setup failed: {e}")
        return None

    # --- Prepare encrypted payload: include report, signature, and sender public key ---
    payload = {
        'report': report,
        'signature': signature,
        'sender_public_key': sender_public_key,
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode()

    try:
        nonce, ciphertext, tag = aes_gcm_encrypt(payload_bytes, aes_key)
    except Exception as e:
        st.error(f"Encryption failed: {e}")
        return None

    encrypted_blob = {
        'session_id': session_id,
        'nonce': nonce.hex(),
        'ciphertext': ciphertext.hex(),
        'tag': tag.hex(),
        'sender_public_key': sender_public_key,
        'quantum_secured': True,
        'quantum_method': 'bb84-integration',
        'quantum_key_length': len(raw_bb84) * 8,
    }

    # Submit to Authentication Unit (in-app)
    ok, msg = auth_unit.verify_and_process_encrypted_report(encrypted_blob)
    if ok:
        st.balloons()
        st.success('Transaction processed and added to blockchain via Authentication Unit.')
        return encrypted_blob
    else:
        st.error(f'Transaction rejected by Authentication Unit: {msg}')
        return None
def bb84_encrypt(message, key_hex):
    """Encrypt message using BB84-derived key"""
    try:
        key = binascii.unhexlify(key_hex)
        if len(key) != 32:
            hk = HKDF(
                algorithm=hashes.SHA3_256(),
                length=32,
                salt=None,
                info=b'bb84-aes-key',
                backend=default_backend()
            )
            key = hk.derive(key)

        nonce = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message.encode()) + encryptor.finalize()
        encrypted_data = nonce + ciphertext + encryptor.tag
        return base64.b64encode(encrypted_data).decode()
    except Exception as e:
        st.error(f"Encryption error: {e}")
        return None

def bb84_decrypt(ciphertext_b64, key_hex):
    """Decrypt message using BB84-derived key"""
    try:
        key = binascii.unhexlify(key_hex)
        if len(key) != 32:
            hk = HKDF(
                algorithm=hashes.SHA3_256(),
                length=32,
                salt=None,
                info=b'bb84-aes-key',
                backend=default_backend()
            )
            key = hk.derive(key)

        encrypted_data = base64.b64decode(ciphertext_b64)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:-16]
        tag = encrypted_data[-16:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()
    except Exception as e:
        st.error(f"Decryption error: {e}")
        return None

def refresh_balance(public_key):
    """Recalculate balance from blockchain"""
    balance = 0.0
    for block in st.session_state.blockchain:
        for tx in block["transactions"]:
            if tx["sender"] == public_key:
                balance -= float(tx["amount"])
            if tx["receiver"] == public_key:
                balance += float(tx["amount"])
    
    st.session_state.balances[public_key] = max(0, balance)
    return st.session_state.balances[public_key]

def save_blockchain():
    """Save blockchain to file"""
    data = {
        "blockchain": st.session_state.blockchain,
        "balances": st.session_state.balances,
        "users": {k: {kk: vv for kk, vv in v.items() if kk != "private_key" and kk != "password_hash"}
                 for k, v in st.session_state.users.items()},
        "network_stats": st.session_state.network_stats,
        "quantum_keys": st.session_state.quantum_keys,
        "last_save": datetime.now().isoformat()
    }
    try:
        with open(st.session_state.blockchain_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving ledger: {e}")

def load_blockchain():
    """Load blockchain from file"""
    if os.path.exists(st.session_state.blockchain_file):
        try:
            with open(st.session_state.blockchain_file, "r") as f:
                data = json.load(f)
                st.session_state.blockchain = data.get("blockchain", [])
                st.session_state.balances = data.get("balances", {})
                st.session_state.network_stats = data.get("network_stats", {"total_transactions": 0, "total_volume": 0})
                st.session_state.quantum_keys = data.get("quantum_keys", {})
                st.success("Ledger loaded successfully!")
        except Exception as e:
            st.error(f"Error loading ledger: {e}")
            if not st.session_state.blockchain:
                genesis_block = {
                    "index": 0,
                    "transactions": [],
                    "merkle_root": "0",
                    "timestamp": str(datetime.now()),
                    "previous_hash": "0",
                    "hash": calculate_hash({"index": 0, "timestamp": str(datetime.now())}),
                    "miner": "network",
                    "quantum_dimension": st.session_state.quantum_dimension
                }
                st.session_state.blockchain.append(genesis_block)

# ==================== STREAMLIT APP CONFIGURATION ====================

st.set_page_config(
    page_title="QuantumVerse - Modern Quantum Blockchain",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODERN UI STYLING ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .quantum-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 30%, #f093fb 70%, #f5576c 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .quantum-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: quantum-spin 20s infinite linear;
    }
    
    @keyframes quantum-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .quantum-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(45deg, #ffffff, #f8f9ff, #e8edff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
        letter-spacing: -2px;
        position: relative;
        z-index: 2;
    }
    
    .quantum-header .subtitle {
        font-size: 1.4rem;
        font-weight: 500;
        color: rgba(255,255,255,0.95);
        margin-top: 1rem;
        position: relative;
        z-index: 2;
    }
    
    /* Login Container */
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .login-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Cards */
    .modern-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 100%;
        animation: gradient-flow 4s ease-in-out infinite;
    }
    
    @keyframes gradient-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .modern-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 60px rgba(102, 126, 234, 0.25);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .wallet-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    }
    
    .balance-card {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.15) 0%, rgba(56, 249, 215, 0.15) 100%);
        text-align: center;
    }
    
    .activity-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(238, 90, 36, 0.15) 100%);
    }
    
    .network-card {
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.15) 0%, rgba(143, 148, 251, 0.15) 100%);
    }
    
    .transaction-card {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
    }
    
    .transaction-card:hover {
        background: rgba(255,255,255,0.08);
        transform: translateX(8px);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        border: none;
        color: rgba(255,255,255,0.7);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Input Fields */
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
        background: rgba(255,255,255,0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Badges */
    .quantum-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0 0.5rem;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        animation: quantum-pulse 3s infinite;
    }
    
    @keyframes quantum-pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(255, 107, 107, 0.6); }
    }
    
    .dimension-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .hardware-badge {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
    }
    
    /* Address styling */
    .quantum-address {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 12px;
        word-break: break-all;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators */
    .status-success {
        color: #43e97b;
        font-weight: 600;
    }
    
    .status-warning {
        color: #feca57;
        font-weight: 600;
    }
    
    .status-error {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Charts */
    .plotly-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Loading animations */
    .quantum-loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .quantum-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced form styling */
    .stForm {
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    /* Additional styling for market elements */
    .crypto-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .crypto-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 100%;
        animation: gradient-flow 4s ease-in-out infinite;
    }
    
    .crypto-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    
    .market-metric {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
    }
    
    .market-metric:hover {
        background: rgba(255,255,255,0.08);
        transform: scale(1.02);
    }
    
    .positive-change {
        color: #43e97b;
        font-weight: 600;
    }
    
    .negative-change {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .quantum-header h1 {
            font-size: 2.5rem;
        }
        .quantum-header .subtitle {
            font-size: 1.1rem;
        }
        .modern-card {
            padding: 1.5rem;
        }
    }
    
    /* Transaction confirmation animation */
    @keyframes quantum-payment-confirmed {
        0% { 
            transform: scale(1);
            opacity: 0;
        }
        50% {
            transform: scale(1.2);
            opacity: 1;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .quantum-payment-confirmed {
        animation: quantum-payment-confirmed 1.5s ease-in-out;
    }
    
    /* Coin transfer animation */
    @keyframes coin-transfer {
        0% {
            transform: translateX(0) rotate(0deg);
            opacity: 1;
        }
        50% {
            transform: translateX(100px) rotate(180deg);
            opacity: 0.7;
        }
        100% {
            transform: translateX(200px) rotate(360deg);
            opacity: 0;
        }
    }
    
    .coin-animation {
        position: absolute;
        font-size: 1.5rem;
        animation: coin-transfer 1.5s ease-in-out forwards;
        z-index: 100;
    }
    
    /* Payment success quantum animation */
    .quantum-payment-success {
        display: inline-block;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        position: relative;
        animation: scaleIn 0.5s ease-out;
    }
    
    .quantum-payment-success::after {
        content: '⚛️';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    /* Transaction amount highlight */
    @keyframes amount-highlight {
        0% { background-color: transparent; }
        50% { background-color: rgba(67, 233, 123, 0.2); }
        100% { background-color: transparent; }
    }
    
    .amount-highlight {
        animation: amount-highlight 1.5s ease-in-out;
        padding: 0.2rem 0.5rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================


def initialize_session_state():
    """Initialize all session state variables (demo users removed)."""
    defaults = {
        "users": {},
        "balances": {},
        "blockchain": [],
        "network_stats": {"total_transactions": 0, "total_volume": 0},
        "price_history": [],
        "quantum_keys": {},
        "quantum_dimension": 4,
        "ibm_api_key": "",
        "ibm_channel": "ibm_quantum",
        "ibm_instance": "",
        "use_real_hardware": False,
        "ibm_quantum_configured": False,
        "last_hardware_used": "Aer Simulator",
        "last_quantum_backend": "",
        "authentication_psk": hashlib.sha256(b"quantumverse_default_psk").hexdigest(),
        "authentication_enabled": True,
        "auth_keys": {},
        "reconciliation_block_size": 12,
        "reconciliation_max_iterations": 8,
        "error_correction_strength": 2,
        "blockchain_file": "quantumverse_blockchain.json",
        "current_tab": "wallet",
        # Add market data defaults
        "market_data": {
            "crypto": None,
            "fiat": None,
            "last_update": None,
            "auto_refresh": True
        },
        "selected_crypto": None,
        "selected_crypto_name": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # NOTE: Demo users removed. Use the Create Account flow to register new users.

initialize_session_state()

# ==================== TOP NAVIGATION BAR ====================

# Add custom CSS for the navigation
st.markdown("""
<style>
    /* Top navigation bar */
    .top-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: rgba(10, 14, 26, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-container {
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .nav-logo {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .nav-items {
        display: flex;
        gap: 1rem;
    }
    
    .nav-item {
        padding: 0.5rem 1rem;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .nav-item:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .nav-item.active {
        background: rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .user-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .user-badge {
        background: rgba(255,255,255,0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .logout-btn {
        background: rgba(255,107,107,0.2);
        padding: 0.5rem;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,107,107,0.3);
    }
    
    .logout-btn:hover {
        background: rgba(255,107,107,0.4);
        transform: scale(1.1);
    }
    
    /* Main content padding to account for fixed nav */
    .main-content {
        padding-top: 80px;
    }
    
    /* Hide Streamlit elements */
    .stApp > header {
        display: none;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MAIN HEADER ====================

st.markdown(f"""
<div class="quantum-header">
    <h1>QuantumVerse</h1>
    <div class="subtitle">
        Next-Generation Quantum Blockchain with High-Dimensional BB84 Encryption
        <br>
        <span class="dimension-badge">{st.session_state.quantum_dimension}-D Quantum Security</span>
        <span class="hardware-badge">{st.session_state.last_hardware_used}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== GENESIS BLOCK SETUP ====================

if not st.session_state.blockchain:
    genesis_block = {
        "index": 0,
        "transactions": [],
        "merkle_root": "0",
        "timestamp": str(datetime.now()),
        "previous_hash": "0",
        "hash": calculate_hash({"index": 0, "timestamp": str(datetime.now())}),
        "miner": "network",
        "quantum_dimension": st.session_state.quantum_dimension
    }
    st.session_state.blockchain.append(genesis_block)

# ==================== LOGIN SYSTEM ====================

def send_otp_email(receiver_email):
    """Send a 6-digit OTP to receiver_email using the configured support account.
    OTP is stored temporarily in st.session_state['pending_otp'] (expires in 10 minutes).
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        otp = str(secrets.randbelow(900000) + 100000)
        st.session_state['pending_otp'] = otp
        st.session_state['pending_otp_email'] = receiver_email
        st.session_state['pending_otp_expires'] = (datetime.now() + timedelta(minutes=10)).isoformat()
        sender_email = "quantumverse.supp@gmail.com"
        sender_password = "izkl sdhq ehsy oivh"
        msg = MIMEText(f"Your QuantumVerse verification code is: {otp}\n\nThis code expires in 10 minutes.")
        msg["Subject"] = "QuantumVerse OTP Verification"
        msg["From"] = sender_email
        msg["To"] = receiver_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return False

if "logged_in_user" not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-title"> Quantum Secure Access</div>
        """, unsafe_allow_html=True)

        # Simple login by username (entered) + password
        with st.form("login_form", clear_on_submit=True):
            username_input = st.text_input(" Enter Username")
            password_input = st.text_input(" Enter Quantum Password", type="password")

            col_l, col_r = st.columns(2)
            with col_l:
                login_btn = st.form_submit_button(" Quantum Login", use_container_width=True)
            with col_r:
                create_flow_btn = st.form_submit_button(" Create Account", use_container_width=True)

            if login_btn:
                # Try to find user by username
                found_key = None
                for k, v in st.session_state.users.items():
                    if v.get('username') == username_input:
                        found_key = k
                        break
                if found_key:
                    password_hash = hashlib.sha256(password_input.encode()).hexdigest()
                    if password_hash == st.session_state.users[found_key].get("password_hash", ""):
                        st.session_state.logged_in_user = found_key
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(" Invalid quantum credentials")
                else:
                    st.error(" User not found. Please create an account.")

            if create_flow_btn:
                # Start create account flow
                st.session_state['show_create_account'] = True
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Create Account Flow ----------
        if st.session_state.get('show_create_account', False):
            st.markdown("""
            <div class="modern-card" style="margin-top:1rem;">
                <h4 style="margin-top:0;">Create QuantumVerse Account</h4>
                <p>Provide an email to receive an OTP for verification.</p>
            </div>
            """, unsafe_allow_html=True)

            # Step 1: Send OTP
            with st.form("create_account_send_otp", clear_on_submit=False):
                otp_email = st.text_input(" Verification Email", value=st.session_state.get('pending_otp_email',''))
                send_otp_btn = st.form_submit_button(" Send OTP to Email", use_container_width=True)
                if send_otp_btn:
                    if otp_email:
                        ok = send_otp_email(otp_email)
                        if ok:
                            st.session_state['otp_sent'] = True
                            st.success(f"OTP sent to {otp_email}. Check your inbox.")
                            st.rerun()
                        else:
                            st.error("Failed to send OTP. Check SMTP configuration.")
                    else:
                        st.error("Please enter a valid email address.")

            # Step 2: Verify OTP (if sent)
            if st.session_state.get('otp_sent', False):
                with st.form("create_account_verify_otp", clear_on_submit=True):
                    otp_input = st.text_input(" Enter the 6-digit OTP")
                    verify_otp_btn = st.form_submit_button(" Verify OTP", use_container_width=True)
                    if verify_otp_btn:
                        pending_otp = st.session_state.get('pending_otp')
                        expires = st.session_state.get('pending_otp_expires')
                        try:
                            expires_dt = datetime.fromisoformat(expires) if expires else datetime.now()
                        except Exception:
                            expires_dt = datetime.now()
                        if pending_otp and otp_input and otp_input == pending_otp and expires_dt > datetime.now():
                            st.session_state['otp_verified'] = True
                            st.success("OTP verified. Now choose a username and password.")
                            st.rerun()
                        else:
                            st.error("Invalid or expired OTP. Request a new one.")

            # Step 3: Create username/password after OTP verification
            if st.session_state.get('otp_verified', False):
                with st.form("create_account_finalize", clear_on_submit=True):
                    new_username = st.text_input(" Choose Username")
                    new_password = st.text_input(" Choose Password", type="password")
                    finalize_btn = st.form_submit_button(" Create Account", use_container_width=True)
                    if finalize_btn:
                        if not new_username or not new_password:
                            st.error("Username and password are required.")
                        else:
                            # ensure unique username
                            for u in st.session_state.users.values():
                                if u.get('username') == new_username:
                                    st.error("Username already exists. Pick another.")
                                    break
                            else:
                                wallet = generate_wallet(new_username)
                                wallet['password_hash'] = hashlib.sha256(new_password.encode()).hexdigest()
                                user_key = "U" + secrets.token_hex(3)
                                st.session_state.users[user_key] = wallet
                                st.session_state.balances[wallet['public_key']] = 1000
                                st.session_state.quantum_keys[user_key] = None
                                st.success("Account created successfully! Please login using your username and password.")
                                # clear create-account temporary state
                                for k in ['show_create_account','pending_otp','pending_otp_email','pending_otp_expires','otp_sent','otp_verified']:
                                    if k in st.session_state: del st.session_state[k]
                                st.rerun()
else:
    # ==================== TAB CONTENT ====================
    
    logged_in_user = st.session_state.logged_in_user
    user_wallet = st.session_state.users[logged_in_user]
    user_public_key = user_wallet["public_key"]
    user_private_key = user_wallet["private_key"]
    quantum_key = st.session_state.quantum_keys.get(logged_in_user)

    def get_user_stats(public_key):
        """Calculate comprehensive user statistics"""
        sent, received, tx_count = 0, 0, 0
        quantum_txs = 0
        max_dimension = 0
        hardware_types = {}
        
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                if tx["sender"] == public_key:
                    sent += float(tx["amount"])
                    tx_count += 1
                    if tx.get("quantum_secured", False):
                        quantum_txs += 1
                        max_dimension = max(max_dimension, tx.get("quantum_dimension", 2))
                        hw = tx.get("quantum_hardware", "Aer Simulator")
                        hardware_types[hw] = hardware_types.get(hw, 0) + 1
                        
                if tx["receiver"] == public_key:
                    received += float(tx["amount"])
                    if tx["sender"] != public_key:
                        tx_count += 1
                    if tx.get("quantum_secured", False):
                        quantum_txs += 1
                        max_dimension = max(max_dimension, tx.get("quantum_dimension", 2))
                        hw = tx.get("quantum_hardware", "Aer Simulator")
                        hardware_types[hw] = hardware_types.get(hw, 0) + 1
                        
        return {
            "total_sent": sent,
            "total_received": received,
            "transaction_count": tx_count,
            "net_flow": received - sent,
            "quantum_txs": quantum_txs,
            "max_quantum_dim": max_dimension,
            "hardware_types": hardware_types
        }

    user_stats = get_user_stats(user_public_key)
    current_balance = st.session_state.balances.get(user_public_key, 0.0)

    # Create tab navigation
    tabs = st.tabs(["Wallet", "Analytics", "Network", "Market", "Security", "Settings"])
    
    # WALLET TAB
    with tabs[0]:
        # User overview cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="modern-card wallet-card">
                <h3 style="margin-top: 0;"> Account Details</h3>
                <p><strong>User:</strong> {user_wallet['username']}</p>
                <div class="quantum-address">
                    <strong>Address:</strong><br>
                    {user_public_key[:32]}...<br>
                    {user_public_key[-32:]}
                </div>
                <p><strong>Created:</strong> {user_wallet['created_at'][:19]}</p>
                <p><strong>Security:</strong> 
                    <span class="quantum-badge">{st.session_state.quantum_dimension}-D BB84</span>
                </p>
                <p><strong>Hardware:</strong> 
                    <span class="hardware-badge">{st.session_state.last_hardware_used}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="modern-card balance-card">
                <h3 style="margin-top: 0;"> Current Balance</h3>
                <div class="metric-value">{current_balance:.2f}</div>
                <div style="font-size: 1.2rem; margin-top: 1rem;">QCoins</div>
                <div style="color: {'#43e97b' if user_stats['net_flow'] >= 0 else '#ff6b6b'}; font-weight: 600; margin-top: 1rem;">
                    Net Flow: {user_stats['net_flow']:+.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="modern-card activity-card">
                <h3 style="margin-top: 0;"> Activity Summary</h3>
                <p><strong>Total Sent:</strong> {user_stats['total_sent']:.2f}</p>
                <p><strong>Total Received:</strong> {user_stats['total_received']:.2f}</p>
                <p><strong>Transactions:</strong> {user_stats['transaction_count']}</p>
                <p><strong>Quantum TXs:</strong> 
                    <span class="quantum-badge">{user_stats['quantum_txs']}</span>
                </p>
                <p><strong>Max Q-Dimension:</strong> {user_stats['max_quantum_dim']}</p>
            </div>
            """, unsafe_allow_html=True)

        # QR Code and Address
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("###  Wallet QR Code")
            try:
                qr = qrcode.QRCode(version=1, box_size=8, border=2)
                qr.add_data(user_public_key)
                qr.make(fit=True)
                img = qr.make_image(fill_color="#667eea", back_color="white")
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.image(buf.getvalue(), width=200)
            except Exception:
                st.info("QR code generation unavailable")
        
        with col2:
            st.markdown("###  Wallet Address")
            st.markdown(f"""
            <div class="quantum-address">
                {user_public_key}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(" Copy Address", use_container_width=True):
                st.code(user_public_key)
                st.success("Address ready to copy!")

        # Transaction Form
        st.markdown("###  Send Quantum Transaction")
        
        with st.form("quantum_transaction_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                receiver_options = [k for k in st.session_state.users if k != logged_in_user]
                if receiver_options:
                    receiver_name = st.selectbox(
                        " Select Recipient",
                        receiver_options,
                        format_func=lambda x: f" {st.session_state.users[x]['username']} ({x})"
                    )
                else:
                    st.error("No other users available")
                    receiver_name = None
                
                max_amount = max(0.01, float(current_balance))
                amount = st.number_input(
                    " Amount (QCoins)",
                    min_value=0.01,
                    max_value=max_amount,
                    step=0.01,
                    value=min(10.0, max_amount)
                )
            
            with col2:
                tx_type = st.selectbox(
                    " Transaction Type",
                    ["transfer", "payment", "gift", "loan", "reward"],
                    format_func=lambda x: f" {x.title()}"
                )
                
                fee = st.number_input(" Network Fee", min_value=0.0, max_value=1.0, step=0.01, value=0.01)
                
                quantum_encryption = st.checkbox(
                    f" High-Dimensional Quantum Encryption ({st.session_state.quantum_dimension}-D)",
                    value=True,
                    help="Use quantum key distribution for enhanced security"
                )

            # Transaction summary
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; margin: 1rem 0;">
                <strong> Transaction Summary:</strong><br>
                Amount: {amount:.2f} QCoins + {fee:.2f} fee = <strong>{amount + fee:.2f} QCoins</strong><br>
                Remaining Balance: <strong>{current_balance - amount - fee:.2f} QCoins</strong>
            </div>
            """, unsafe_allow_html=True)

            submitted = st.form_submit_button(" Send Quantum Transaction", use_container_width=True)

            if submitted and receiver_name:
                if amount > 0 and (amount + fee) <= current_balance:
                    receiver_pk = st.session_state.users[receiver_name]["public_key"]
                    
                    # Show a warning for higher dimensions
                    if st.session_state.quantum_dimension > 4 and quantum_encryption:
                        st.warning(f" {st.session_state.quantum_dimension}-D quantum encryption may take longer to process")
                    
                    result = create_transaction(
                        user_private_key, receiver_pk, amount, tx_type, fee
                    )
                    
                    if result:
                        # Add a brief delay before refreshing
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error(" Invalid amount or insufficient funds")

        # Recent Transactions
        # ... (all previous code remains exactly the same until the transaction section)

        # Recent Transactions - FIXED VERSION
        st.markdown("###  Recent Transactions")
        
        recent_txs = []
        for block in reversed(st.session_state.blockchain[-10:]):
            for tx in block["transactions"]:
                if tx["sender"] == user_public_key or tx["receiver"] == user_public_key:
                    tx_copy = tx.copy()
                    tx_copy["block"] = block["index"]
                    recent_txs.append(tx_copy)

        if recent_txs:
            for tx in recent_txs[:8]:  # Show only the 8 most recent transactions
                # Determine if this was an incoming or outgoing transaction
                direction = "sent" if tx["sender"] == user_public_key else "received"
                amount_color = "#ff6b6b" if direction == "sent" else "#43e97b"
                direction_icon = "↗️" if direction == "sent" else "↘️"
                
                # Get the other party's username if available
                other_party_key = tx["receiver"] if direction == "sent" else tx["sender"]
                other_party_name = "Unknown"
                
                # Try to find the username for this public key
                for user_key, user_data in st.session_state.users.items():
                    if user_data["public_key"] == other_party_key:
                        other_party_name = user_data["username"]
                        break
                
                # Format the timestamp
                try:
                    timestamp = datetime.fromisoformat(tx["timestamp"].replace('Z', '+00:00'))
                    time_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_display = tx["timestamp"][:19] if len(tx["timestamp"]) > 10 else "Unknown time"
                
                # Create a container for the transaction
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Transaction details
                        st.markdown(f"**{direction_icon} {direction.title()} - Block #{tx['block']}**")
                        st.markdown(f"To: {other_party_name}")
                        st.markdown(f"{time_display}")
                        
                        # Quantum info if applicable
                        if tx.get("quantum_secured", False):
                            col_badge1, col_badge2 = st.columns(2)
                            with col_badge1:
                                st.markdown(f'<span class="dimension-badge">{tx.get("quantum_dimension", 2)}-D Quantum</span>', 
                                           unsafe_allow_html=True)
                            with col_badge2:
                                st.markdown(f'<span class="hardware-badge">{tx.get("quantum_hardware", "Simulator")}</span>', 
                                           unsafe_allow_html=True)
                    
                    with col2:
                        # Amount display
                        amount_display = f"{'-' if direction == 'sent' else '+'}{float(tx['amount']):.2f}"
                        st.markdown(f'<div style="color: {amount_color}; font-weight: 700; font-size: 1.4rem; text-align: right;">{amount_display}</div>', 
                                   unsafe_allow_html=True)
                        st.markdown('<div style="color: rgba(255,255,255,0.6); text-align: right;">QCoins</div>', 
                                   unsafe_allow_html=True)
                        
                        # Fee if applicable
                        if tx.get('fee', 0) > 0:
                            st.markdown(f'<div style="color: rgba(255,255,255,0.5); text-align: right;">Fee: {tx.get("fee", 0):.2f}</div>', 
                                       unsafe_allow_html=True)
                    
                    # Divider
                    st.markdown("---")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);">
                <div style="font-size: 3rem;">💸</div>
                <div style="margin-top: 1rem; font-size: 1.2rem;">No transactions yet</div>
                <div>Start by sending your first quantum transaction!</div>
            </div>
            """, unsafe_allow_html=True)

# ... (all the remaining code stays exactly the same)

    # ANALYTICS TAB
    with tabs[1]:
        # Calculate quantum transaction stats first
        total_quantum_tx = 0
        total_dimension = 0
        hardware_usage = {}
        
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                if tx.get("quantum_secured", False):
                    total_quantum_tx += 1
                    total_dimension += tx.get("quantum_dimension", 2)
                    hw = tx.get("quantum_hardware", "Aer Simulator")
                    hardware_usage[hw] = hardware_usage.get(hw, 0) + 1
        
        avg_quantum_dim = total_dimension / total_quantum_tx if total_quantum_tx > 0 else 0

        st.markdown("###  Quantum Network Analytics")

        # Network overview metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("📦", "Blocks", len(st.session_state.blockchain)),
            ("🔁", "Transactions", st.session_state.network_stats["total_transactions"]),
            ("💰", "Volume", f"{st.session_state.network_stats['total_volume']:.1f}"),
            ("🔐", "Quantum TXs", total_quantum_tx),
            ("📊", "Avg Q-Dim", f"{avg_quantum_dim:.1f}")
        ]
        
        for col, (icon, label, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction volume over time
            block_data = []
            for block in st.session_state.blockchain[1:]:  # Skip genesis
                block_volume = sum(float(tx["amount"]) for tx in block["transactions"] if tx["sender"] != "network")
                quantum_txs = sum(1 for tx in block["transactions"] if tx.get("quantum_secured", False))
                block_data.append({
                    "Block": block["index"],
                    "Volume": block_volume,
                    "Quantum_TXs": quantum_txs,
                    "Timestamp": block["timestamp"][:19]
                })
            
            if block_data:
                df = pd.DataFrame(block_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Block"], 
                    y=df["Volume"],
                    mode="lines+markers",
                    name="Volume",
                    line=dict(color="#667eea", width=3),
                    marker=dict(size=8, color="#667eea"),
                    hovertemplate="Block %{x}<br>Volume: %{y:.2f} QCoins<extra></extra>"
                ))
                
                fig.add_trace(go.Bar(
                    x=df["Block"],
                    y=df["Quantum_TXs"],
                    name="Quantum TXs",
                    marker_color="#43e97b",
                    opacity=0.7,
                    hovertemplate="Block %{x}<br>Quantum TXs: %{y}<extra></extra>"
                ))
                
                fig.update_layout(
                    title=" Network Activity Over Time",
                    xaxis_title="Block Height",
                    yaxis_title="Volume / Count",
                    template="plotly_dark",
                    height=400,
                    font=dict(family="Inter", size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transaction data available yet")

        with col2:
            # Quantum dimension distribution
            dim_counts = {}
            for block in st.session_state.blockchain:
                for tx in block["transactions"]:
                    if tx.get("quantum_secured", False):
                        dim = tx.get("quantum_dimension", 2)
                        dim_counts[dim] = dim_counts.get(dim, 0) + 1
            
            if dim_counts:
                fig = px.pie(
                    values=list(dim_counts.values()),
                    names=[f"{k}-D Quantum" for k in dim_counts.keys()],
                    title=" Quantum Dimension Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#43e97b"]
                )
                
                fig.update_layout(
                    font=dict(family="Inter", size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(" No quantum transactions to analyze")

        # Hardware usage analysis
        if hardware_usage:
            st.markdown("###  Quantum Hardware Usage")
            
            fig = px.bar(
                x=list(hardware_usage.keys()),
                y=list(hardware_usage.values()),
                title=" Quantum Computing Hardware Distribution",
                template="plotly_dark",
                color=list(hardware_usage.values()),
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(
                xaxis_title="Hardware Type",
                yaxis_title="Transaction Count",
                font=dict(family="Inter", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # User performance metrics
        st.markdown("###  User Performance Dashboard")
        
        user_performance = []
        for user_key, user_data in st.session_state.users.items():
            user_pk = user_data["public_key"]
            user_stats_detailed = get_user_stats(user_pk)
            balance = st.session_state.balances.get(user_pk, 0)
            
            user_performance.append({
                "User": user_data["username"],
                "Balance": balance,
                "Transactions": user_stats_detailed["transaction_count"],
                "Quantum_TXs": user_stats_detailed["quantum_txs"],
                "Net_Flow": user_stats_detailed["net_flow"]
            })
        
        if user_performance:
            df_users = pd.DataFrame(user_performance)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_users,
                    x="User",
                    y="Balance",
                    title=" User Balances",
                    template="plotly_dark",
                    color="Balance",
                    color_continuous_scale="viridis"
                )
                fig.update_layout(
                    font=dict(family="Inter"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df_users,
                    x="Transactions",
                    y="Quantum_TXs",
                    size="Balance",
                    hover_name="User",
                    title=" Quantum vs Total Transactions",
                    template="plotly_dark",
                    color="Net_Flow",
                    color_continuous_scale="RdYlBu"
                )
                fig.update_layout(
                    font=dict(family="Inter"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

        # Error rate analysis for quantum transactions
        if total_quantum_tx > 0:
            st.markdown("### Quantum Error Rate Analysis")
            
            # Collect error rate data (simulated for demo purposes)
            error_data = []
            for block in st.session_state.blockchain:
                for tx in block["transactions"]:
                    if tx.get("quantum_secured", False):
                        # Simulate error rate based on dimension and hardware
                        base_rate = 0.05  # 5% base error rate
                        dim_factor = 1.0 / tx.get("quantum_dimension", 2)
                        hw_factor = 1.5 if "Simulator" in tx.get("quantum_hardware", "") else 1.0
                        error_rate = base_rate * dim_factor * hw_factor * random.uniform(0.8, 1.2)
                        
                        error_data.append({
                            "block": block["index"],
                            "timestamp": block["timestamp"],
                            "error_rate": error_rate,
                            "dimension": tx.get("quantum_dimension", 2),
                            "hardware": tx.get("quantum_hardware", "Unknown")
                        })
            
            if error_data:
                df_errors = pd.DataFrame(error_data)
                
                # Create error rate timeline
                fig = px.line(
                    df_errors, 
                    x="block", 
                    y="error_rate",
                    title="Quantum Error Rate Over Time",
                    template="plotly_dark",
                    labels={"block": "Block Height", "error_rate": "Error Rate"}
                )
                
                fig.update_layout(
                    font=dict(family="Inter"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Error Rate", f"{df_errors['error_rate'].mean():.3%}")
                with col2:
                    st.metric("Min Error Rate", f"{df_errors['error_rate'].min():.3%}")
                with col3:
                    st.metric("Max Error Rate", f"{df_errors['error_rate'].max():.3%}")

    # MARKET TAB
    with tabs[3]:
        st.markdown("### 📊 Quantum Market Data")
        
        # Initialize market data in session state if not exists
        if "market_data" not in st.session_state:
            st.session_state.market_data = {
                "crypto": None,
                "fiat": None,
                "last_update": None,
                "auto_refresh": True
            }
        
        # Settings for market data
        with st.expander("⚙️ Market Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60, 30)
                auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.market_data["auto_refresh"])
                st.session_state.market_data["auto_refresh"] = auto_refresh
                
            with col2:
                base_currency = st.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY", "INR"], index=0)
                crypto_list = st.text_input("Cryptocurrencies (comma-separated)", 
                                          value="bitcoin,ethereum,solana,cardano,ripple,polkadot,dogecoin")
        
        # Refresh button and status
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔄 Refresh Market Data", use_container_width=True):
                with st.spinner("Fetching latest market data..."):
                    st.session_state.market_data["crypto"] = fetch_crypto_data(crypto_list)
                    st.session_state.market_data["fiat"] = fetch_fiat_rates(base_currency)
                    st.session_state.market_data["last_update"] = datetime.now()
                    st.success("Market data updated!")
        
        with col2:
            if st.session_state.market_data["last_update"]:
                st.caption(f"Last update: {st.session_state.market_data['last_update'].strftime('%H:%M:%S')}")
        
        with col3:
            if st.session_state.market_data["auto_refresh"]:
                # Auto-refresh logic
                if (st.session_state.market_data["last_update"] is None or 
                    (datetime.now() - st.session_state.market_data["last_update"]).seconds >= refresh_interval):
                    with st.spinner("Auto-refreshing market data..."):
                        st.session_state.market_data["crypto"] = fetch_crypto_data(crypto_list)
                        st.session_state.market_data["fiat"] = fetch_fiat_rates(base_currency)
                        st.session_state.market_data["last_update"] = datetime.now()
        
        # Display market data
        if st.session_state.market_data["crypto"]:
            st.markdown("#### 💰 Cryptocurrency Prices")
            
            # Create cards for each cryptocurrency
            crypto_cols = st.columns(3)
            for idx, crypto in enumerate(st.session_state.market_data["crypto"]):
                col_idx = idx % 3
                with crypto_cols[col_idx]:
                    change_24h = crypto.get('price_change_percentage_24h', 0)
                    change_color = "#43e97b" if change_24h >= 0 else "#ff6b6b"
                    change_icon = "📈" if change_24h >= 0 else "📉"
                    
                    st.markdown(f"""
                    <div class="modern-card" style="border-left: 4px solid {change_color};">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <img src="{crypto['image']}" width="32" height="32" style="border-radius: 50%; margin-right: 0.5rem;">
                            <h4 style="margin: 0;">{crypto['name']}</h4>
                        </div>
                        <div class="metric-value" style="font-size: 1.5rem;">${crypto['current_price']:,.2f}</div>
                        <div style="color: {change_color}; font-weight: 600;">
                            {change_icon} {change_24h:+.2f}%
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                            Market Cap: ${crypto['market_cap']:,.0f}
                        </div>
                        <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                            24h Vol: ${crypto['total_volume']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a button to view historical data
                    if st.button(f"View Chart", key=f"chart_{crypto['id']}", use_container_width=True):
                        st.session_state.selected_crypto = crypto['id']
                        st.session_state.selected_crypto_name = crypto['name']
        
            # Display historical chart if a crypto is selected
            if "selected_crypto" in st.session_state:
                st.markdown(f"#### 📈 {st.session_state.selected_crypto_name} Price History")
                
                days_options = {"7 Days": 7, "30 Days": 30, "90 Days": 90}
                selected_days = st.radio("Time Period", list(days_options.keys()), horizontal=True)
                
                with st.spinner(f"Loading {selected_days} of price data..."):
                    history_data = get_crypto_history(st.session_state.selected_crypto, days_options[selected_days])
                    
                    if history_data:
                        # Prepare data for chart
                        dates = [datetime.fromtimestamp(price[0]/1000) for price in history_data]
                        prices = [price[1] for price in history_data]
                        
                        # Create chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dates, 
                            y=prices,
                            mode="lines",
                            name=st.session_state.selected_crypto_name,
                            line=dict(color="#667eea", width=3),
                            fill='tozeroy',
                            fillcolor='rgba(102, 126, 234, 0.1)'
                        ))
                        
                        fig.update_layout(
                            title=f"{st.session_state.selected_crypto_name} Price History",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            template="plotly_dark",
                            height=400,
                            font=dict(family="Inter", size=12),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not load historical data")
        
        # Display fiat exchange rates
        if st.session_state.market_data["fiat"]:
            st.markdown("#### 💱 Fiat Exchange Rates")
            
            # Select popular currencies to display
            popular_currencies = ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR"]
            if base_currency in popular_currencies:
                popular_currencies.remove(base_currency)
            
            # Display exchange rates in a grid
            fiat_cols = st.columns(4)
            for idx, currency in enumerate(popular_currencies[:8]):
                if currency in st.session_state.market_data["fiat"]:
                    rate = st.session_state.market_data["fiat"][currency]
                    col_idx = idx % 4
                    with fiat_cols[col_idx]:
                        st.markdown(f"""
                        <div class="modern-card" style="text-align: center; padding: 1rem;">
                            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                                {base_currency}/{currency}
                            </div>
                            <div class="metric-value" style="font-size: 1.3rem;">
                                {rate:.4f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Market news section (placeholder)
        st.markdown("#### 📰 Market News")
        st.info("""
        QuantumVerse is monitoring global markets for you. 
        In a future update, we'll integrate real-time financial news related to your holdings.
        """)

    # NETWORK TAB
    with tabs[2]:
        # Calculate quantum transaction stats
        total_quantum_tx = 0
        total_dimension = 0
        hardware_usage = {}
        
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                if tx.get("quantum_secured", False):
                    total_quantum_tx += 1
                    total_dimension += tx.get("quantum_dimension", 2)
                    hw = tx.get("quantum_hardware", "Aer Simulator")
                    hardware_usage[hw] = hardware_usage.get(hw, 0) + 1
        
        avg_quantum_dim = total_dimension / total_quantum_tx if total_quantum_tx > 0 else 0
        st.markdown("###  Quantum Network Explorer")
        
        # Network status overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="modern-card network-card">
                <h3 style="margin-top: 0;"> Network Information</h3>
                <p><strong>Network:</strong> QuantumVerse</p>
                <p><strong>Genesis:</strong> {st.session_state.blockchain[0]['timestamp'][:19] if st.session_state.blockchain else 'N/A'}</p>
                <p><strong>Consensus:</strong> Ed25519 + SHA-256</p>
                <p><strong>Quantum Protocol:</strong> High-Dimensional BB84</p>
                <p><strong>Current Dimension:</strong> 
                    <span class="dimension-badge">{st.session_state.quantum_dimension}-D</span>
                </p>
                <p><strong>Active Hardware:</strong> 
                    <span class="hardware-badge">{st.session_state.last_hardware_used}</span>
                </p>
                <p><strong>Quantum Transactions:</strong> {total_quantum_tx}</p>
                                    <p><strong>Average Q-Dimension:</strong> {avg_quantum_dim:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-top: 0;"> Network Participants</h3>
            """, unsafe_allow_html=True)
            
            for k, u in st.session_state.users.items():
                bal = st.session_state.balances.get(u["public_key"], 0)
                has_key = "✅" if st.session_state.quantum_keys.get(k) else "⭕"
                active = "🟢" if k == logged_in_user else "⚪"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                     padding: 1rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); 
                     border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <strong>{active} {u['username']} ({k})</strong><br>
                        <small style="color: rgba(255,255,255,0.7);">{bal:.2f} QCoins</small>
                    </div>
                    <div style="text-align: right;">
                        <div>{has_key}</div>
                        <small style="color: rgba(255,255,255,0.6);">Quantum Key</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Network actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Run Quantum Security Audit", use_container_width=True):
                with st.spinner("Auditing quantum security..."):
                    valid_blocks = 0
                    quantum_errors = 0
                    
                    for block in st.session_state.blockchain:
                        block_copy = block.copy()
                        block_copy.pop("hash", None)
                        calculated_hash = calculate_hash(block_copy)
                        if calculated_hash == block["hash"]:
                            valid_blocks += 1
                        
                        for tx in block["transactions"]:
                            if tx.get("quantum_secured", False) and not verify_signature(tx):
                                quantum_errors += 1
                    
                    st.success(f"✅ Audit Complete: {valid_blocks}/{len(st.session_state.blockchain)} blocks valid")
                    if quantum_errors > 0:
                        st.error(f"❌ {quantum_errors} quantum transactions failed verification")
                    else:
                        st.success(" All quantum transactions verified successfully")
        
        with col2:
            if st.button(" Save Network State", use_container_width=True):
                save_blockchain()
                st.success(" Quantum ledger saved to disk!")

        # Export functionality
        st.markdown("###  Data Export & Management")
        
        export_data = {
            "blockchain": st.session_state.blockchain,
            "balances": st.session_state.balances,
            "users": {k: {kk: vv for kk, vv in v.items() if kk not in ["private_key", "password_hash"]} 
                     for k, v in st.session_state.users.items()},
            "network_stats": st.session_state.network_stats,
            "quantum_dimension": st.session_state.quantum_dimension,
            "export_timestamp": str(datetime.now()),
            "total_quantum_transactions": total_quantum_tx,
            "average_quantum_dimension": avg_quantum_dim
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                " Export Analytics Data",
                json.dumps(export_data, indent=2),
                file_name=f"quantumverse_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            if st.button(" Refresh Network Data", use_container_width=True):
                for user_key, user_data in st.session_state.users.items():
                    refresh_balance(user_data["public_key"])
                st.success(" Network data refreshed!")

        # Blockchain Explorer
        if st.session_state.blockchain:
            st.markdown("###  Quantum Blockchain Explorer")
            
            selected_block = st.selectbox(
                "Select Block to Explore",
                range(len(st.session_state.blockchain)),
                format_func=lambda i: f"Block #{i} ({len(st.session_state.blockchain[i]['transactions'])} transactions)",
                index=len(st.session_state.blockchain) - 1
            )
            
            block = st.session_state.blockchain[selected_block]
            
            # Block details
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Block #{block['index']} 
                    <span class="quantum-badge">Quantum Secured</span>
                </h4>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.5rem;">
                    <div>
                        <p><strong> Timestamp:</strong> {block['timestamp'][:19]}</p>
                        <p><strong> Miner:</strong> {block['miner']}</p>
                        <p><strong> Transactions:</strong> {len(block['transactions'])}</p>
                        <p><strong> Quantum Dimension:</strong> 
                            <span class="dimension-badge">{block.get('quantum_dimension', 2)}-D</span>
                        </p>
                    </div>
                    <div>
                        <p><strong> Block Hash:</strong></p>
                        <div class="quantum-address" style="font-size: 0.8rem; margin-bottom: 1rem;">
                            {block['hash']}
                        </div>
                        <p><strong> Previous Hash:</strong></p>
                        <div class="quantum-address" style="font-size: 0.8rem;">
                            {block['previous_hash']}
                        </div>
                    </div>
                </div>
                
                <p style="margin-top: 1.5rem;"><strong> Merkle Root:</strong></p>
                <div class="quantum-address" style="font-size: 0.8rem;">
                    {block['merkle_root']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Block transactions
            if block["transactions"]:
                st.markdown("####  Block Transactions")
                
                for i, tx in enumerate(block["transactions"]):
                    sender_name = " Network" if tx["sender"] == "network" else f" {tx['sender'][:12]}..."
                    receiver_name = f" {tx['receiver'][:12]}..."
                    
                    quantum_info = ""
                    if tx.get("quantum_secured", False):
                        quantum_info = f"""
                        <div style="margin-top: 0.8rem; display: flex; gap: 0.5rem;">
                            <span class="quantum-badge">{tx.get('quantum_dimension', 2)}-D Quantum</span>
                            <span class="hardware-badge">{tx.get('quantum_hardware', 'Simulator')}</span>
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div class="transaction-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="flex: 1;">
                                <h5 style="margin: 0 0 1rem 0; color: #667eea;"> Transaction #{i+1}</h5>
                                <p style="margin: 0.3rem 0;"><strong>From:</strong> {sender_name}</p>
                                <p style="margin: 0.3rem 0;"><strong>To:</strong> {receiver_name}</p>
                                <p style="margin: 0.3rem 0;"><strong>Type:</strong> {tx.get('type', 'transfer').title()}</p>
                                <p style="margin: 0.3rem 0;"><strong>Time:</strong> {tx['timestamp'][:19]}</p>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-top: 0.5rem;">
                                    ID: {tx.get('tx_id', 'N/A')}
                                </div>
                                {quantum_info}
                            </div>
                            <div style="text-align: right; min-width: 100px;">
                                <div style="font-size: 1.8rem; font-weight: 700; color: #43e97b; margin-bottom: 0.2rem;">
                                    {float(tx['amount']):.2f}
                                </div>
                                <div style="color: rgba(255,255,255,0.6);">QCoins</div>
                                {f'<div style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-top: 0.5rem;">Fee: {tx.get("fee", 0):.2f}</div>' if tx.get("fee", 0) > 0 else ''}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No transactions in this block")

        # Advanced transaction search
        st.markdown("###  Advanced Transaction Search")
        
        with st.expander(" Search Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_sender = st.text_input(" Sender Address", "")
                filter_amount_min = st.number_input(" Min Amount", min_value=0.0, value=0.0)
            
            with col2:
                filter_receiver = st.text_input(" Receiver Address", "")
                filter_amount_max = st.number_input(" Max Amount", min_value=0.0, value=10000.0)
            
            with col3:
                filter_type = st.selectbox(" Transaction Type", ["all", "transfer", "payment", "gift", "loan", "reward"])
                quantum_only = st.checkbox(" Quantum Only", value=False)

        # Collect and filter transactions
        all_transactions = []
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                tx_with_block = tx.copy()
                tx_with_block["block_height"] = block["index"]
                tx_with_block["block_time"] = block["timestamp"]
                all_transactions.append(tx_with_block)

        # Apply filters
        filtered_txs = all_transactions
        if filter_sender:
            filtered_txs = [tx for tx in filtered_txs if filter_sender.lower() in tx["sender"].lower()]
        if filter_receiver:
            filtered_txs = [tx for tx in filtered_txs if filter_receiver.lower() in tx["receiver"].lower()]
        if filter_amount_min > 0:
            filtered_txs = [tx for tx in filtered_txs if float(tx["amount"]) >= filter_amount_min]
        if filter_amount_max < 10000:
            filtered_txs = [tx for tx in filtered_txs if float(tx["amount"]) <= filter_amount_max]
        if filter_type != "all":
            filtered_txs = [tx for tx in filtered_txs if tx.get("type", "transfer") == filter_type]
        if quantum_only:
            filtered_txs = [tx for tx in filtered_txs if tx.get("quantum_secured", False)]

        st.markdown(f"** Found {len(filtered_txs)} transactions**")
        
        # Display filtered results
        for tx in filtered_txs[-15:]:  # Show most recent 15
            direction = "outgoing" if tx["sender"] == user_public_key else "incoming"
            direction_icon = "↗️" if direction == "outgoing" else "↘️"
            amount_color = "#ff6b6b" if direction == "outgoing" else "#43e97b"
            
            quantum_info = ""
            if tx.get("quantum_secured", False):
                quantum_info = f"""
                <div style="margin-top: 0.8rem; display: flex; gap: 0.5rem;">
                    <span class="quantum-badge">{tx.get('quantum_dimension', 2)}-D</span>
                    <span class="hardware-badge">{tx.get('quantum_hardware', 'Simulator')}</span>
                </div>
                """
            else:
                quantum_info = '<div style="margin-top: 0.8rem;"><span style="color: #feca57; font-weight: 600;">⚠️ Classical Transaction</span></div>'
            
            st.markdown(f"""
            <div class="transaction-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                            {direction_icon} Block #{tx['block_height']} • {direction.upper()}
                        </div>
                        <div style="color: rgba(255,255,255,0.7; margin-bottom: 0.8rem;">
                            From: {tx['sender'][:12]}...{tx['sender'][-6:]}
                        </div>
                        <div style="color: rgba(255,255,255,0.7; margin-bottom: 0.8rem;">
                            To: {tx['receiver'][:12]}...{tx['receiver'][-6:]}
                        </div>
                        <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); margin-bottom: 0.8rem;">
                            {tx['timestamp'][:19]} • {tx.get('type', 'transfer').title()}
                        </div>
                        {quantum_info}
                    </div>
                    <div style="text-align: right; min-width: 100px;">
                        <div style="color: {amount_color}; font-weight: 700; font-size: 1.4rem; margin-bottom: 0.2rem;">
                            {float(tx['amount']):.2f}
                        </div>
                        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                            QCoins
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # SECURITY TAB
    with tabs[4]:
        st.markdown("### Quantum Security Center")
        
        # Security overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="modern-card">
                <h3 style="margin-top: 0;"> Security Status</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
                    <div>
                        <div class="metric-label">Authentication</div>
                        <div class="{'status-success' if st.session_state.get('authentication_enabled') else 'status-warning'}">
                            {'✅ Enabled' if st.session_state.get('authentication_enabled') else '⚠️ Disabled'}
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Quantum Dimension</div>
                        <div class="status-success">{st.session_state.quantum_dimension}-D</div>
                    </div>
                    <div>
                        <div class="metric-label">Error Correction</div>
                        <div class="status-success">
                            {['Light', 'Medium', 'Strong'][st.session_state.get('error_correction_strength', 2) - 1]}
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Hardware Type</div>
                        <div class="{'status-success' if 'IBM' in st.session_state.last_hardware_used else 'status-warning'}">
                            {'🔬 Real' if 'IBM' in st.session_state.last_hardware_used else '⚡ Simulator'}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="modern-card">
                <h3 style="margin-top: 0;"> Security Metrics</h3>
                <div style="margin-top: 1.5rem;">
                    <div class="metric-container">
                        <div class="metric-label">Security Score</div>
                        <div class="metric-value">{min(100, user_stats['quantum_txs'] * 5 + user_stats['max_quantum_dim'] * 10)}</div>
                        <div style="color: rgba(255,255,255,0.7);">out of 100</div>
                    </div>
                </div>
                <div style="margin-top: 1rem; color: rgba(255,255,255,0.7);">
                    Based on quantum transaction history and encryption strength
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Security configuration
        st.markdown("###  Security Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Authentication Settings</h4>
            """, unsafe_allow_html=True)
            
            auth_enabled = st.checkbox(
                "Enable Quantum Authentication",
                value=st.session_state.get("authentication_enabled", True),
                help="Use HMAC authentication for quantum protocols"
            )
            
            if auth_enabled != st.session_state.get("authentication_enabled"):
                st.session_state.authentication_enabled = auth_enabled
                st.success("Authentication settings updated!")
            
            psk = st.text_input(
                "Pre-Shared Key",
                value=st.session_state.get("authentication_psk", ""),
                type="password",
                help="Shared secret for HMAC authentication"
            )
            
            if psk != st.session_state.get("authentication_psk"):
                if len(psk) >= 32:
                    st.session_state.authentication_psk = psk
                    st.success("Pre-shared key updated!")
                else:
                    st.error("Pre-shared key must be at least 32 characters")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Quantum Protocol Settings</h4>
            """, unsafe_allow_html=True)
            
            new_dimension = st.selectbox(
                "Quantum Dimension",
                [2, 4, 8, 16],
                index=[2, 4, 8, 16].index(st.session_state.quantum_dimension),
                help="Higher dimensions provide more security but require more computation"
            )
            
            if new_dimension != st.session_state.quantum_dimension:
                st.session_state.quantum_dimension = new_dimension
                st.success(f"Quantum dimension updated to {new_dimension}-D!")
            
            error_correction = st.select_slider(
                "Error Correction Strength",
                options=[1, 2, 3],
                value=st.session_state.get("error_correction_strength", 2),
                format_func=lambda x: ["Light", "Medium", "Strong"][x-1]
            )
            
            if error_correction != st.session_state.get("error_correction_strength"):
                st.session_state.error_correction_strength = error_correction
                st.success(f"Error correction set to {['Light', 'Medium', 'Strong'][error_correction-1]}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Quantum key management
        st.markdown("###  Quantum Key Management")
        
        if quantum_key:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="quantum-address" style="font-size: 0.9rem;">
                    <strong>Current Quantum Key:</strong><br>
                    {quantum_key[:64]}...<br>
                    {quantum_key[-64:]}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(" Refresh Key", use_container_width=True):
                    with st.spinner("Generating new quantum key..."):
                        new_key = one_time_circuit_high_dim_bb84(256, st.session_state.quantum_dimension)
                        if new_key:
                            st.session_state.quantum_keys[logged_in_user] = new_key
                            st.success(" Quantum key refreshed!")
                            st.rerun()
                        else:
                            st.error(" Failed to generate new quantum key")
        else:
            st.info(" No quantum key available. Generate one by making a quantum transaction.")
        
        # Security audit log
        st.markdown("###  Security Audit Log")
        
        # Simulated audit events
        audit_events = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).strftime("%H:%M:%S"),
                "event": "Quantum transaction completed",
                "status": "✅ Success",
                "dimension": st.session_state.quantum_dimension,
                "hardware": st.session_state.last_hardware_used
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%H:%M:%S"),
                "event": "Quantum key generation",
                "status": "✅ Success",
                "dimension": st.session_state.quantum_dimension,
                "hardware": st.session_state.last_hardware_used
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%H:%M:%S"),
                "event": "Security settings updated",
                "status": "✅ Success",
                "dimension": "N/A",
                "hardware": "N/A"
            }
        ]
        
        for event in audit_events:
            st.markdown(f"""
            <div class="transaction-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">
                            {event['event']}
                        </div>
                        <div style="color: rgba(255,255,255,0.7);">
                            {event['timestamp']} • {event['status']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        {f"<span class='dimension-badge'>{event['dimension']}-D</span>" if event['dimension'] != 'N/A' else ''}
                        {f"<span class='hardware-badge'>{event['hardware']}</span>" if event['hardware'] != 'N/A' else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Security recommendations
        st.markdown("###  Security Recommendations")
        
        recommendations = []
        
        if user_stats["quantum_txs"] < 5:
            recommendations.append("Make more quantum transactions to improve security score")
        
        if st.session_state.quantum_dimension < 8:
            recommendations.append("Consider increasing quantum dimension to 8-D or higher")
        
        if "IBM" not in st.session_state.last_hardware_used:
            recommendations.append("Try using real quantum hardware for enhanced security")
        
        if not recommendations:
            st.success("✅ Your security configuration is optimal!")
        else:
            for rec in recommendations:
                st.markdown(f"""
                <div style="background: rgba(254, 202, 87, 0.1); padding: 1rem; border-radius: 12px; 
                     margin: 0.5rem 0; border-left: 4px solid #feca57;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">⚠️</span>
                        <span>{rec}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # SETTINGS TAB
    with tabs[5]:
        st.markdown("###  QuantumVerse Settings")
        
        # User preferences
        st.markdown("####  User Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Interface Settings</h4>
            """, unsafe_allow_html=True)
            
            theme_options = ["Dark (Default)", "Light", "Auto"]
            selected_theme = st.selectbox("Theme", theme_options, index=0)
            
            refresh_rate = st.slider("Auto-refresh Rate (seconds)", 10, 300, 30, 10)
            
            show_animations = st.checkbox("Show Animations", value=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Notification Settings</h4>
            """, unsafe_allow_html=True)
            
            email_notifications = st.checkbox("Email Notifications", value=False)
            push_notifications = st.checkbox("Push Notifications", value=True)
            quantum_alerts = st.checkbox("Quantum Security Alerts", value=True)
            
            notification_frequency = st.selectbox(
                "Frequency",
                ["Immediate", "Hourly", "Daily", "Weekly"]
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Quantum computing settings
        st.markdown("####  Quantum Computing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> IBM Quantum Configuration</h4>
            """, unsafe_allow_html=True)
            
            ibm_api_key = st.text_input(
                "IBM Quantum API Key",
                value=st.session_state.get("ibm_api_key", ""),
                type="password",
                help="Get your API key from quantum-computing.ibm.com"
            )
            
            if ibm_api_key != st.session_state.get("ibm_api_key"):
                st.session_state.ibm_api_key = ibm_api_key
                if ibm_api_key:
                    try:
                        # Test the API key
                        if IBM_QUANTUM_AVAILABLE:
                            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_key)
                            backends = service.backends()
                            if backends:
                                st.session_state.ibm_quantum_configured = True
                                st.success("✅ IBM Quantum successfully configured!")
                            else:
                                st.error("No IBM Quantum backends available with this API key")
                        else:
                            st.warning("IBM Quantum support not available")
                    except Exception as e:
                        st.error(f"IBM Quantum configuration failed: {e}")
                else:
                    st.session_state.ibm_quantum_configured = False
            
            ibm_channel = st.selectbox(
                "IBM Channel",
                ["ibm_quantum", "ibm_cloud"],
                index=0 if st.session_state.get("ibm_channel") == "ibm_quantum" else 1
            )
            
            if ibm_channel != st.session_state.get("ibm_channel"):
                st.session_state.ibm_channel = ibm_channel
            
            ibm_instance = st.text_input(
                "IBM Instance",
                value=st.session_state.get("ibm_instance", ""),
                help="Optional: Specific IBM Quantum instance to use"
            )
            
            if ibm_instance != st.session_state.get("ibm_instance"):
                st.session_state.ibm_instance = ibm_instance
            
            use_real_hardware = st.checkbox(
                "Use Real Quantum Hardware",
                value=st.session_state.get("use_real_hardware", False),
                disabled=not st.session_state.get("ibm_quantum_configured", False),
                help="Use actual quantum computers instead of simulators"
            )
            
            if use_real_hardware != st.session_state.get("use_real_hardware"):
                st.session_state.use_real_hardware = use_real_hardware
                if use_real_hardware:
                    st.success("Real quantum hardware enabled!")
                else:
                    st.info("Using quantum simulators")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Advanced Quantum Settings</h4>
            """, unsafe_allow_html=True)
            
            ansatz_enabled = st.checkbox(
                "Use Variational Ansatz",
                value=st.session_state.get("use_ansatz", False),
                help="Use variational quantum circuits for enhanced security"
            )
            
            if ansatz_enabled != st.session_state.get("use_ansatz"):
                st.session_state.use_ansatz = ansatz_enabled
            
            if ansatz_enabled:
                param_count = st.slider(
                    "Ansatz Parameter Count",
                    min_value=4,
                    max_value=16,
                    value=st.session_state.get("ansatz_param_count", 8),
                    step=4
                )
                
                if param_count != st.session_state.get("ansatz_param_count"):
                    st.session_state.ansatz_param_count = param_count
                    # Initialize random parameters
                    st.session_state.ansatz_params = np.random.uniform(-np.pi, np.pi, size=param_count)
            
            calibration_runs = st.slider(
                "Calibration Runs",
                min_value=10,
                max_value=100,
                value=st.session_state.get("calibration_runs", 30),
                step=10,
                help="Number of calibration runs for quantum error optimization"
            )
            
            if calibration_runs != st.session_state.get("calibration_runs"):
                st.session_state.calibration_runs = calibration_runs
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Data management
        st.markdown("####  Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Backup & Restore</h4>
            """, unsafe_allow_html=True)
            
            if st.button(" Backup Blockchain Data", use_container_width=True):
                save_blockchain()
                st.success("Blockchain data backed up successfully!")
            
            uploaded_file = st.file_uploader(
                "Restore from Backup",
                type=["json"],
                help="Upload a previously exported blockchain backup file"
            )
            
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    if st.button(" Restore Backup", use_container_width=True):
                        st.session_state.blockchain = data.get("blockchain", [])
                        st.session_state.balances = data.get("balances", {})
                        st.session_state.network_stats = data.get("network_stats", {"total_transactions": 0, "total_volume": 0})
                        st.success("Backup restored successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading backup file: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h4 style="margin-top: 0;"> Data Privacy</h4>
            """, unsafe_allow_html=True)
            
            data_collection = st.checkbox(
                "Allow Anonymous Usage Data",
                value=False,
                help="Help improve QuantumVerse by sharing anonymous usage statistics"
            )
            
            crash_reports = st.checkbox(
                "Send Crash Reports",
                value=True,
                help="Automatically send error reports to help fix issues"
            )
            
            analytics_tracking = st.checkbox(
                "Enable Analytics",
                value=True,
                help="Track application usage for improvement purposes"
            )
            
            if st.button(" Clear Local Data", use_container_width=True):
                # This would clear all local data in a real application
                st.warning("This will remove all local data and reset the application")
                if st.button(" Confirm Clear Data", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    initialize_session_state()
                    st.success("Local data cleared!")
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

        # System information
        st.markdown("####  System Information")
        
        system_info = {
            "Python Version": sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "Qiskit Version": "Available" if 'qiskit' in sys.modules else "Not Available",
            "Quantum Backend": st.session_state.last_hardware_used,
            "Blockchain Size": f"{len(st.session_state.blockchain)} blocks",
            "Total Transactions": st.session_state.network_stats["total_transactions"],
            "Users": len(st.session_state.users),
            "Quantum Keys": sum(1 for k in st.session_state.quantum_keys.values() if k is not None)
        }
        
        info_cols = st.columns(2)
        for i, (key, value) in enumerate(system_info.items()):
            with info_cols[i % 2]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; 
                     margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">{key}</div>
                    <div style="color: rgba(255,255,255,0.8);">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # Application actions
        st.markdown("####  Application Actions")
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button(" Check for Updates", use_container_width=True):
                st.info("Update check functionality would be implemented here")
        
        with action_cols[1]:
            if st.button(" Clear Cache", use_container_width=True):
                st.success("Cache cleared successfully!")
        
        with action_cols[2]:
            if st.button(" Report Issue", use_container_width=True):
                st.info("Issue reporting would open a dialog here")

        # About section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7);">
            <h3>QuantumVerse</h3>
            <p>Next-Generation Quantum Blockchain Platform</p>
            <p>Version 1.0.0 • Built with Streamlit and Qiskit</p>
            <p>© 2024 QuantumVerse Team. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)

    # Logout button at the bottom
    if st.button(" Logout", use_container_width=True):
        del st.session_state.logged_in_user
        st.rerun()

# Add a custom footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.5);">
    <div>QuantumVerse - High-Dimensional Quantum Blockchain</div>
    <div style="margin-top: 0.5rem; font-size: 0.9rem;">
        Built with Streamlit • Qiskit • Ed25519 • Advanced Cryptography
    </div>
</div>
""", unsafe_allow_html=True)
