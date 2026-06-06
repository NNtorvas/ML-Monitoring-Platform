import sys
from unittest.mock import MagicMock

# Mock evidently submodules so tests can import monitoring.drift_report without
# depending on any particular installed version of evidently. Individual tests
# patch monitoring.drift_report.Report directly for controlled behaviour.
sys.modules.setdefault("evidently.report", MagicMock())
sys.modules.setdefault("evidently.metric_preset", MagicMock())
