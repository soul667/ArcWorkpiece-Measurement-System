import { useEffect, useCallback } from 'react';

/**
 * Custom hook for handling keyboard events
 * @param {Object} params Keyboard event handlers and state
 * @returns {void}
 */
const useKeyboard = ({
  onModeToggle,
  onRegionConfirm,
  onLineUndo,
  lines,
  onAxisModeToggle
}) => {
  // Handle keyboard events
  const handleKeyDown = useCallback((e) => {
    // Mode switching with X/Y/Z keys
    if (e.key.toLowerCase() === 'x') {
      e.preventDefault();
      onAxisModeToggle('x');
      return;
    }
    if (e.key.toLowerCase() === 'y') {
      e.preventDefault();
      onAxisModeToggle('y');
      return;
    }
    if (e.key.toLowerCase() === 'z') {
      e.preventDefault();
      onAxisModeToggle('z');
      return;
    }

    // Region and drawing controls
    if (e.key === 'Enter' && lines.length >= 2) {
      e.preventDefault();
      onRegionConfirm();
    }
    
    if (e.key.toLowerCase() === 'r' && lines.length > 0) {
      e.preventDefault();
      onLineUndo();
    }
    
    if (e.key.toLowerCase() === 'm') {
      e.preventDefault();
      onModeToggle();
    }
  }, [lines, onModeToggle, onRegionConfirm, onLineUndo, onAxisModeToggle]);

  // Setup keyboard event listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
};

export default useKeyboard;
