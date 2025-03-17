/**
 * FilterSection module exports
 */

export { default } from './components/FilterSection';

// Export individual components if needed
export { default as FilterCanvas } from './components/Canvas/FilterCanvas';
export { default as CanvasOverlay } from './components/Canvas/CanvasOverlay';
export { default as ViewControls } from './components/Controls/ViewControls';
export { default as ModeControls } from './components/Controls/ModeControls';
export { default as RegionList } from './components/Regions/RegionList';

// Export hooks
export { default as useCanvas } from './hooks/useCanvas';
export { default as useImage } from './hooks/useImage';
export { default as useRegions } from './hooks/useRegions';
export { default as useKeyboard } from './hooks/useKeyboard';

// Export utils
export * from './utils/canvasUtils';
export * from './utils/imageUtils';
