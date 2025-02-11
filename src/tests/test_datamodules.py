import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


from src.datamodules.module import PointCloudDataModule


def test_pc_datamodule() -> None:
    # Expected sizes
    batch_size = 5
    n_csts = 16
    edge_dim = 6
    node_dim = 3
    high_dim = 2

    # Load the module with some basic configs
    test_module = PointCloudDataModule(
        val_frac=0.1,
        data_conf={
            "path": str(root / "src/tests/tiny_rodem_data"),
            "datasets": {"c0": "tiny_rodem"},
            "n_jets": 10,
            "n_csts": n_csts,
            "coordinates": {
                "edge": [
                    "del_R",
                    "log_kt",
                    "z",
                    "log_m",
                    "psi",
                    "dot_prod",
                ],
                "node": [
                    "del_eta",
                    "del_phi",
                    "log_pt",
                ],
                "high": ["pt", "mass"],
            },
            "min_n_csts": 1,
            "leading": True,
            "recalculate_jet_from_pc": False,
            "incl_substruc": False,
            "del_r_edges": 9999,
            "boost_mopt": 0,
            "augmentation_list": [],
            "augmentation_prob": 0,
        },
        loader_kwargs={
            "pin_memory": True,
            "batch_size": batch_size,
            "num_workers": 1,
            "drop_last": True,
        },
    )

    # Setup for fitting and load the first batches
    test_module.setup("test")
    edges, nodes, high, adjmat, mask, label = next(iter(test_module.test_dataloader()))
    assert list(edges.shape) == [batch_size, n_csts, n_csts, edge_dim]
    assert list(nodes.shape) == [batch_size, n_csts, node_dim]
    assert list(high.shape) == [batch_size, high_dim]
    assert list(adjmat.shape) == [batch_size, n_csts, n_csts]
    assert list(mask.shape) == [batch_size, n_csts]
