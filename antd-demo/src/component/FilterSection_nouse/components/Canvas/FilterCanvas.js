import React, { useRef } from 'react';
import PropTypes from 'prop-types';
import useCanvas from '../../hooks/useCanvas';

const FilterCanvas = ({
  imageRef,
  isFullscreen,
  drawMode,
  xRegions,
  yRegions,
  lines,
  onLineAdd
}) => {
  const canvasRef = useRef(null);
  const { canvasStyle } = useCanvas({
    canvasRef,
    imageRef,
    isFullscreen,
    drawMode,
    xRegions,
    yRegions,
    lines,
    onLineAdd
  });

  return (
    <canvas 
      ref={canvasRef} 
      style={canvasStyle}
    />
  );
};

FilterCanvas.propTypes = {
  imageRef: PropTypes.object.isRequired,
  isFullscreen: PropTypes.bool.isRequired,
  drawMode: PropTypes.oneOf(['x', 'y']).isRequired,
  xRegions: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)).isRequired,
  yRegions: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)).isRequired,
  lines: PropTypes.arrayOf(
    PropTypes.shape({
      x: PropTypes.number,
      y: PropTypes.number
    })
  ).isRequired,
  onLineAdd: PropTypes.func.isRequired
};

export default FilterCanvas;
