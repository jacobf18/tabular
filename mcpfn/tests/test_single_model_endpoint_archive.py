import pytest

import archive_single_model_decoder_endpoint as endpoint


def test_original_concurrent_writer_storage_cushion_remains_unchanged():
    assert endpoint.archive.MINIMUM_FREE == 21 * 1024**3
    assert endpoint.SOURCE.name == "raw_final_update.pt"
    assert endpoint.SOURCE.parent.name == "decoder_repair_v14"
    assert "checkpoint" not in endpoint.SOURCE.name


@pytest.mark.parametrize("state,pid,complete", [("capturing_gradients", 999999999, True), ("training_complete", 1, True),
                                               ("training_complete", 999999999, False)])
def test_live_or_unaudited_source_cannot_be_archived(monkeypatch, state, pid, complete):
    monkeypatch.setattr(endpoint, "read", lambda path: {"state": state, "pid": pid} if path.name == "status.json"
                        else {"complete_training_audit": complete, "through_step": 50})
    with pytest.raises(ValueError, match="terminal fully audited"):
        endpoint.source_record()
