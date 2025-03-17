import React from 'react';
import PropTypes from 'prop-types';
import { Button } from 'antd';
import { FullscreenOutlined, FullscreenExitOutlined } from '@ant-design/icons';

const ViewControls = ({
  currentView,
  isFullscreen,
  onViewChange,
  onFullscreenToggle
}) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <Button.Group>
        <Button 
          onClick={() => onViewChange('xy')}
          type={currentView === 'xy' ? 'primary' : 'default'}
        >
          XY
        </Button>
        <Button 
          onClick={() => onViewChange('xz')}
          type={currentView === 'xz' ? 'primary' : 'default'}
        >
          XZ
        </Button>
        <Button 
          onClick={() => onViewChange('yz')}
          type={currentView === 'yz' ? 'primary' : 'default'}
        >
          YZ
        </Button>
      </Button.Group>
      <Button 
        icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
        onClick={onFullscreenToggle}
      >
        {isFullscreen ? '退出全屏' : '全屏'}
      </Button>
    </div>
  );
};

ViewControls.propTypes = {
  currentView: PropTypes.oneOf(['xy', 'xz', 'yz']).isRequired,
  isFullscreen: PropTypes.bool.isRequired,
  onViewChange: PropTypes.func.isRequired,
  onFullscreenToggle: PropTypes.func.isRequired
};

export default ViewControls;
