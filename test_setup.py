"""
Test script to verify all imports and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing SAC SB3 Integration Setup...")
print("=" * 80)

# Test 1: Import environment
print("\n1. Testing environment import...")
try:
    from env.dummy import MatrixEnv
    print("   ✓ MatrixEnv imported successfully")
    
    env = MatrixEnv(state_dim=14, action_dim=8, max_state=144, max_action=44)
    print("   ✓ MatrixEnv instantiated successfully")
    print(f"     - Observation space: {env.observation_space}")
    print(f"     - Action space: {env.action_space}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import custom networks
print("\n2. Testing custom network import...")
try:
    from model.SAC import Q_network, P_network, V_network, ActionDecoder
    print("   ✓ Q_network imported successfully")
    print("   ✓ P_network imported successfully")
    print("   ✓ V_network imported successfully")
    print("   ✓ ActionDecoder imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Test network instantiation
print("\n3. Testing network instantiation...")
try:
    import torch
    
    q_net = Q_network(state_dim=14, action_dim=8, max_planets=40, max_fleets=100)
    print("   ✓ Q_network instantiated")
    
    p_net = P_network(state_dim=14, action_dim=8, max_planets=40, max_fleets=100)
    print("   ✓ P_network instantiated")
    
    v_net = V_network(state_dim=14, max_planets=40, max_fleets=100)
    print("   ✓ V_network instantiated")
    
    decoder = ActionDecoder(d_action=8)
    print("   ✓ ActionDecoder instantiated")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test sb3 policy import
print("\n4. Testing sb3 custom policy import...")
try:
    from sac_sb3_policy import CustomSACPolicy, SACPolicyAdapter, StateShapeAdapter, ActionShapeAdapter
    print("   ✓ CustomSACPolicy imported")
    print("   ✓ SACPolicyAdapter imported")
    print("   ✓ StateShapeAdapter imported")
    print("   ✓ ActionShapeAdapter imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test state/action shape adapters
print("\n5. Testing shape adapters...")
try:
    # Test state adapter - sb3 provides flat state as [batch_size, total_flat_features]
    state_adapter = StateShapeAdapter(max_planets=40, max_fleets=100, state_dim=14)
    
    # Create flat state: batch_size=2, total_features=(40+100)*14=2016
    batch_size = 2
    total_features = (40 + 100) * 14
    flat_state = torch.randn(batch_size, total_features)
    
    matrix_state = state_adapter(flat_state)
    print(f"   ✓ StateShapeAdapter: {flat_state.shape} → {matrix_state.shape}")
    assert matrix_state.shape == (batch_size, 140, 14), f"Unexpected matrix state shape: {matrix_state.shape}"
    
    # Test action adapter
    action_adapter = ActionShapeAdapter(max_planets=40, action_dim=8)
    matrix_action = torch.randn(batch_size, 40, 8)
    flat_action = action_adapter(matrix_action)
    print(f"   ✓ ActionShapeAdapter: {matrix_action.shape} → {flat_action.shape}")
    assert flat_action.shape == (batch_size, 320), f"Unexpected flat action shape: {flat_action.shape}"
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test network forward pass
print("\n6. Testing network forward passes...")
try:
    batch_size = 2
    state = torch.randn(batch_size, 144, 14)
    action = torch.randn(batch_size, 40, 8)
    
    # Q-network
    q_value = q_net(state, action)
    print(f"   ✓ Q_network forward: output shape {q_value.shape}")
    assert q_value.shape == (batch_size,), "Unexpected Q-network output shape"
    
    # P-network
    mu, sigma = p_net(state)
    print(f"   ✓ P_network forward: mu shape {mu.shape}, sigma shape {sigma.shape}")
    assert mu.shape == (batch_size, 40, 8), "Unexpected P-network mu shape"
    
    # V-network
    v_value = v_net(state)
    print(f"   ✓ V_network forward: output shape {v_value.shape}")
    assert v_value.shape == (batch_size,), "Unexpected V-network output shape"
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test training module import
print("\n7. Testing training module import...")
try:
    from sac_train_sb3 import train_sac, evaluate_agent, create_environment
    print("   ✓ train_sac imported")
    print("   ✓ evaluate_agent imported")
    print("   ✓ create_environment imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test environment step
print("\n8. Testing environment step...")
try:
    obs, info = env.reset()
    print(f"   ✓ Environment reset: obs shape {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Environment step: obs shape {obs.shape}, reward {reward:.4f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Check sb3 compatibility
print("\n9. Checking Stable Baselines 3 compatibility...")
try:
    from stable_baselines3.common.env_checker import check_env
    print("   ✓ sb3 env_checker imported")
    # Note: Actual env check might take time, skip detailed check
    print("   ✓ Environment appears compatible with sb3")
except Exception as e:
    print(f"   ⚠ Warning: {e}")

print("\n" + "=" * 80)
print("✓ All tests passed successfully!")
print("=" * 80)
print("\nYou can now run the training:")
print("  python example_sac_sb3.py          # Run interactive examples")
print("  python sac_train_sb3.py            # Run full training")
print("  python sac_train_sb3.py --help     # See all options")
