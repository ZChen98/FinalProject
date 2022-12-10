"""
Justin Chen
Final Project - Unit Test
"""

from functions import prep_data, visual_data, generate_stat, run_models, PARAM_INTERESTS

def test_prep_data():
    """Test to catch missing values"""
    data = prep_data()
    assert data[PARAM_INTERESTS].isna().sum().all() < 1

def test_visual_data():
    """Test to ensure data visualization
        The joint plot generated will count as 2 figures. Therefore, there should be
        4 figures in total
    """
    fig_num = visual_data()
    assert fig_num == 3

def test_generate_stat():
    """Test for descriptive statistics"""
    stats = generate_stat().loc[['mean', 'std']]
    assert stats[stats == 0].sum().any() is not True

def test_run_models():
    """Test for models for analysis"""
    model_name, score, model = run_models()
    assert score > 0.5
    assert model is not None
    assert model_name is not None
