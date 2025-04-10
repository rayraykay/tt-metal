## Swin_S Model

## Platforms
    WH n150

## Test the Swin_S whole model:
- To run the Swin Transformer model pipeline, run the command: `pytest --disable-warnings tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer.py`

## Test the Swin_S submodules:
- To run the Shifted Window Attention Submodule of Swin_s functional model, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_shifted_window_attention.py`
- To run the MLP submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_mlp.py`
- To run the Swin Transformer Block submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer_block.py`
- To run the PatchMerging submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_patchmerging.py`
