import pytest
from datetime import datetime
from app.graph.state import AgentState, create_initial_state
from app.graph.conditional_edges import (
    should_continue_improvement,
    select_next_action,
    evaluate_stopping_conditions
)


def test_create_initial_state():
    """Test initial state creation."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target",
        problem_type="classification",
        max_iterations=5
    )
    
    assert state['job_id'] == "job_123"
    assert state['dataset_path'] == "/data/test.csv"
    assert state['dataset_id'] == "ds_456"
    assert state['target_column'] == "target"
    assert state['problem_type'] == "classification"
    assert state['iteration_number'] == 0
    assert state['max_iterations'] == 5
    assert state['current_step'] == "dataset_analysis"
    assert state['should_continue'] == True
    assert state['experiments'] == []
    assert isinstance(state['created_at'], datetime)


def test_should_continue_improvement_max_iterations():
    """Test stopping at max iterations."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target",
        max_iterations=5
    )
    state['iteration_number'] = 5
    state['should_continue'] = True
    
    result = should_continue_improvement(state)
    assert result == "stop"


def test_should_continue_improvement_critic_stop():
    """Test stopping when critic says stop."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target"
    )
    state['should_continue'] = False
    state['iteration_number'] = 1
    
    result = should_continue_improvement(state)
    assert result == "stop"


def test_should_continue_improvement_min_improvement():
    """Test stopping when improvement below threshold."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target"
    )
    state['iteration_number'] = 2
    state['should_continue'] = True
    state['improvement_plan'] = {'expected_improvement': 0.005}  # Below 0.01
    
    result = should_continue_improvement(state)
    assert result == "stop"


def test_should_continue_improvement_continue():
    """Test continuing when improvement is good."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target"
    )
    state['iteration_number'] = 1
    state['should_continue'] = True
    state['improvement_plan'] = {'expected_improvement': 0.05}
    
    result = should_continue_improvement(state)
    assert result == "continue"


def test_select_next_action():
    """Test action selection from critic feedback."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target"
    )
    state['critic_feedback'] = {'recommended_action': 'class_weight'}
    
    action = select_next_action(state)
    assert action == 'apply_class_weight'
    
    state['critic_feedback'] = {'recommended_action': 'feature_engineering'}
    action = select_next_action(state)
    assert action == 'engineer_features'
    
    state['critic_feedback'] = {'recommended_action': 'none'}
    action = select_next_action(state)
    assert action == 'stop'


def test_evaluate_stopping_conditions():
    """Test stopping condition evaluation."""
    state = create_initial_state(
        job_id="job_123",
        dataset_path="/data/test.csv",
        dataset_id="ds_456",
        target_column="target",
        max_iterations=5
    )
    
    # Not stopped
    result = evaluate_stopping_conditions(state)
    assert result['should_stop'] == False
    assert result['reasons'] == []
    
    # Max iterations
    state['iteration_number'] = 5
    result = evaluate_stopping_conditions(state)
    assert result['should_stop'] == True
    assert 'Maximum iterations reached' in result['reasons']
    
    # Target reached
    state['iteration_number'] = 1
    state['best_metrics'] = {'f1': 0.96}
    result = evaluate_stopping_conditions(state)
    assert result['should_stop'] == True
    assert any('Target F1 score' in r for r in result['reasons'])
    
    # No improvement
    state['best_metrics'] = {'f1': 0.8}
    state['improvement_plan'] = {'expected_improvement': 0.001}
    result = evaluate_stopping_conditions(state)
    assert result['should_stop'] == True
    assert any('Improvement below threshold' in r for r in result['reasons'])