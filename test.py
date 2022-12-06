"""
Justin Chen
Final Project - Unit Test
"""

import matplotlib.pyplot as plt
from FinalProject import functions as f

def test_prep_data():
    """Test to catch missing values"""
    data = f.prep_data()
    assert data['column'].isna().sum() < 1

def test_visual_data():
    """Test to ensure data visualization"""
    f.visual_data()
    assert plt.gcf().number == 1

def test_generate_stat():
    """Test for descriptive statistics"""
    result = f.generate_stat()
    expected = 6
    assert result == expected

def test_run_models():
    """Test for models for analysis"""
    score = f.run_models()
    assert score > 0.5
