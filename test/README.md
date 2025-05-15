# Kivy Image Editor

A Kivy and Plyer based image editor prototype that allows users to load, manage and edit multiple images with layered management, position adjustment, rotation and transformation.

## Features

- **Multiple Image Loading**: Can load multiple images, each as an independent layer
- **Layer Management**: Support for layer selection, show/hide, delete and order adjustment
- **Image Operations**:
  - Drag to Move: Directly drag images to adjust position
  - Rotation: Rotate images using the center control point
  - Transformation: Transform images using corner control points
    - Non-proportional transformation: Directly drag corner points
    - Proportional scaling: Hold Shift key while dragging corner points

## Usage

1. **Environment Setup**:
   - Ensure Python 3 is installed
   - Install dependencies: `pip install kivy plyer`

2. **Start Application**:
   - Run `python image_editor.py`

3. **Basic Operations**:
   - **Load Image**: Click the "Load Image" button to select one or more image files
   - **Select Layer**: Click on an image to select the current working layer
   - **Move Layer**: Directly drag the image
   - **Rotate Image**: Drag the red control point at the center
   - **Transform Image**:
     - Drag the blue control points at the corners for non-proportional transformation
     - Hold Shift key while dragging control points for proportional scaling
   - **Layer Management**:
     - Move Up/Down: Adjust layer order
     - Show/Hide: Toggle layer visibility
     - Delete: Remove the currently selected layer

## Interface Guide

- **Green Border**: Indicates the currently selected layer
- **Red Center Point**: Used for rotation operations
- **Blue Corner Points**: Used for transformation operations

## Notes

- This prototype is developed with Kivy, supporting cross-platform use
- The file selection feature requires the Plyer library
- Optimized for use on Windows systems

## Planned Features

- Undo/Redo functionality
- Canvas zooming and panning
- More image editing options

## License

MIT License 