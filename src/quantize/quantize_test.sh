# Step 1: Clear the stale cache
rm -rf ~/.cache/torch_extensions/
echo "✓ Cleared cache"

# Step 2: Check what MLC version you have
pip list | grep mlc

# Step 3: If it shows cu128, uninstall and install cu124
pip uninstall mlc-llm-nightly-cu128 mlc-ai-nightly-cu128 -y

pip install https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_llm_nightly_cu124-0.20.dev93-py3-none-manylinux_2_28_x86_64.whl

pip install https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_nightly_cu124-0.20.dev537-py3-none-manylinux_2_28_x86_64.whl

# Step 4: Test
python -c "import mlc_llm; print('✓ Success!')"
