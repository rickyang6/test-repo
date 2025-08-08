from iris import IrisClassifier

def test_load_data_creates_dataframe():
    clf = IrisClassifier()
    df = clf.load_data()

    assert df is not None
    assert "species" in df.columns
    assert len(df) > 0

def test_binary_target_encoding():
    clf = IrisClassifier()
    df = clf.load_data(binary_target="setosa")

    unique_values = df["species"].unique()
    assert set(unique_values) == {0, 1}

def test_default_target_column_used():
    clf = IrisClassifier()
    clf.load_data()
    
    assert clf.target_column in clf.data.columns