# QuantCEUS GUI (PyQt6 MVC)

Initial scaffold mirroring architecture of QuantUS-Plugins GUI.

Structure:
- application_model.py : Unified state & background workers (placeholder CEUS loaders)
- application_controller.py : Stack-based navigation root
- mvc/ : Base MVC abstractions

Next Steps:
1. Implement CEUS-specific loader registry (similar to get_scan_loaders) and replace placeholder import in `application_model.py`.
2. Create image_loading/ module with view coordinator & controller adapted from QuantUS `image_loading` components.
3. Add additional workflow stages (curve loading, paramap generation, visualization) following segmentation workflow patterns from QuantUS if needed.
4. Wire entrypoints to actual CEUS pipeline functions in `src/full_workflow.py` or `src/entrypoints.py`.
5. Package GUI entrypoint in pyproject via console_scripts.

Run Dev (when controllers implemented):
```bash
python -m src.gui.run
```

Styling replicates dark theme from existing QuantUS GUI.
