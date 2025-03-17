import { useState, useCallback } from 'react';
import { message } from 'antd';
import axios from '../../../utils/axios';
import { transformRegions } from '../utils/imageUtils';

/**
 * Custom hook for managing regions and related operations
 * @param {Object} params Region parameters
 * @returns {Object} Region state and handlers
 */
const useRegions = ({
  currentView,
  coordinateRanges,
  showPointCloud,
  xMode,
  yMode,
  zMode,
  onSuccess = () => {}
}) => {
  const [xRegions, setXRegions] = useState([]);
  const [yRegions, setYRegions] = useState([]);
  const [lines, setLines] = useState([]);
  const [drawMode, setDrawMode] = useState('x');

  // Add new line
  const handleLineAdd = useCallback((newLine) => {
    setLines(prev => [...prev, newLine]);
  }, []);

  // Undo last line
  const handleLineUndo = useCallback(() => {
    setLines(prev => prev.slice(0, -1));
  }, []);

  // Toggle draw mode
  const handleModeToggle = useCallback(() => {
    setDrawMode(prev => prev === 'x' ? 'y' : 'x');
    setLines([]);
  }, []);

  // Confirm region
  const handleRegionConfirm = useCallback(() => {
    if (lines.length < 2) return;

    if (drawMode === 'x') {
      const xCoords = lines.map(line => line.x).sort((a, b) => a - b);
      const start = xCoords[0];
      const end = xCoords[xCoords.length - 1];
      if (start !== undefined && end !== undefined) {
        setXRegions(prev => [...prev, [start, end]]);
      }
    } else {
      const yCoords = lines.map(line => line.y).sort((a, b) => a - b);
      const start = yCoords[0];
      const end = yCoords[yCoords.length - 1];
      if (start !== undefined && end !== undefined) {
        setYRegions(prev => [...prev, [start, end]]);
      }
    }
    setLines([]);
  }, [lines, drawMode]);

  // Delete region
  const handleRegionDelete = useCallback((index, type) => {
    if (type === 'x') {
      setXRegions(prev => {
        const newRegions = [...prev];
        newRegions.splice(index, 1);
        return newRegions;
      });
    } else {
      setYRegions(prev => {
        const newRegions = [...prev];
        newRegions.splice(index, 1);
        return newRegions;
      });
    }
  }, []);

  // Crop operation
  const handleCrop = useCallback(async () => {
    if (xRegions.length === 0 && yRegions.length === 0) {
      message.warning('请先选择裁剪区域');
      return;
    }

    try {
      const regions = transformRegions({
        xRegions,
        yRegions,
        currentView,
        coordinateRanges
      });

      await axios.post('/api/point-cloud/crop', {
        regions,
        modes: {
          x_mode: xMode,
          y_mode: yMode,
          z_mode: zMode
        },
        settings: {
          show: showPointCloud
        }
      });

      setXRegions([]);
      setYRegions([]);
      setLines([]);
      message.success('裁剪成功');
      onSuccess();
    } catch (error) {
      console.error('Crop error:', error);
      message.error('裁剪操作失败');
    }
  }, [
    xRegions,
    yRegions,
    currentView,
    coordinateRanges,
    xMode,
    yMode,
    zMode,
    showPointCloud,
    onSuccess
  ]);

  // Reset all regions
  const resetRegions = useCallback(() => {
    setXRegions([]);
    setYRegions([]);
    setLines([]);
  }, []);

  return {
    xRegions,
    yRegions,
    lines,
    drawMode,
    handleLineAdd,
    handleLineUndo,
    handleModeToggle,
    handleRegionConfirm,
    handleRegionDelete,
    handleCrop,
    resetRegions
  };
};

export default useRegions;
