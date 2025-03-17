import React from 'react';
import PropTypes from 'prop-types';
import { Typography } from 'antd';

const { Text } = Typography;

const overlayStyles = {
  info: {
    position: 'fixed',
    bottom: '20px',
    left: '20px',
    zIndex: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: '10px',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
  },
  controls: {
    position: 'fixed',
    top: '20px',
    right: '20px',
    zIndex: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: '10px',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  }
};

const CanvasOverlay = ({
  drawMode,
  isFullscreen,
  children
}) => {
  if (!isFullscreen) return null;

  return (
    <>
      <div style={overlayStyles.info}>
        <Text type="secondary">
          当前模式: {drawMode === 'x' ? '垂直选择' : '水平选择'} | 
          按键说明: M切换模式，Enter确认区域，R撤销上一步，XYZ切换保留/移除
        </Text>
      </div>
      <div style={overlayStyles.controls}>
        {children}
      </div>
    </>
  );
};

CanvasOverlay.propTypes = {
  drawMode: PropTypes.oneOf(['x', 'y']).isRequired,
  isFullscreen: PropTypes.bool.isRequired,
  children: PropTypes.node
};

export default CanvasOverlay;
