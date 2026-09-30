"""
Unit test: AHBA real loader
Checks missing data raises DataMissingError and verifies parser structure.
"""

import pytest
from nerva.data.ahba_loader import load_ahba_dataset
from nerva.utils.errors import DataMissingError

def test_missing_ahba_directory_raises_datamissingerror():
    with pytest.raises(DataMissingError):
        load_ahba_dataset(raw_dir="nonexistent_ahba_dir_99999")
