import React, { useRef, useState, useCallback } from 'react';
import { Card, Row, Col, Button, message } from 'antd';
import { ScissorOutlined } from '@ant-design/icons';
import axios from '../../../utils/axios';

import FilterCanvas from './Canvas/FilterCanvas';
import CanvasOverlay from './Canvas/CanvasOverlay';
import ViewControls from './Controls/ViewControls';
import ModeControls from './Controls/ModeControls';
import RegionList from './Regions/RegionList';

import useCanvas from '../hooks/useCanvas';
import useImage from '../hooks/useImage';
import useRegions from '../hooks/useRegions';
import useKeyboard from '../hooks/useKeyboard';

const FilterSection = () => {
  // Refs
  const imageRef = useRef(new Image());
  
  // States
  const [currentView, setCurrentView] = useState('xy');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [coordinateRanges, setCoordinateRanges] = useState({});
  const [xMode, setXMode] = useState('keep');
  const [yMode, setYMode] = useState('keep');
  const [zMode, setZMode] = useState('keep');
  const [nbNeighbors, setNbNeighbors] = useState(100);
  const [stdRatio, setStdRatio] = useState(0.5);
  const [showPointCloud, setShowPointCloud] = useState(false);

  // Callbacks
  const handleImageLoad = useCallback(() => {
    // Handle image load complete
  }, []);

  const handleCropSuccess = useCallback(() => {
    reloadImage();
  }, []);

  // State management with custom hooks
  const { canvasStyle } = useCanvas({
    imageRef,
    isFullscreen,
    onSuccess: handleImageLoad
  });

  const {
    loading: imageLoading,
    error: imageError,
    reloadImage
  } = useImage({
    imageRef,
    currentView,
    onCoordinateRangesUpdate: setCoordinateRanges,
    onImageLoad: handleImageLoad
  });

  const {
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
  } = useRegions({
    currentView,
    coordinateRanges,
    showPointCloud,
    xMode,
    yMode,
    zMode,
    onSuccess: handleCropSuccess
  });

  // Keyboard event handling
  useKeyboard({
    onModeToggle: handleModeToggle,
    onRegionConfirm: handleRegionConfirm,
    onLineUndo: handleLineUndo,
    lines,
    onAxisModeToggle: handleAxisModeToggle
  });

  // View change handler
  const handleViewChange = (view) => {
    setCurrentView(view);
    resetRegions();
  };

  // Fullscreen toggle handler
  const handleFullscreenToggle = async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
      setIsFullscreen(!isFullscreen);
    } catch (err) {
      console.error('Fullscreen error:', err);
      message.error('全屏模式切换失败');
    }
  };

  // Mode change handler
  const handleAxisModeToggle = (axis) => {
    switch (axis) {
      case 'x':
        setXMode(prev => prev === 'keep' ? 'remove' : 'keep');
        break;
      case 'y':
        setYMode(prev => prev === 'keep' ? 'remove' : 'keep');
        break;
      case 'z':
        setZMode(prev => prev === 'keep' ? 'remove' : 'keep');
        break;
    }
  };

  // Denoise handler
  const handleDenoise = async () => {
    try {
      await axios.post('/api/point-cloud/denoise', {
        nb_neighbors: nbNeighbors,
        std_ratio: stdRatio,
        settings: {
          show: showPointCloud
        }
      });
      message.success('去噪处理成功');
      reloadImage();
    } catch (error) {
      console.error('Denoise error:', error);
      message.error('去噪处理失败');
    }
  };

  return (
    <Card bordered={false}>
      <Row gutter={[16, 16]}>
        {!isFullscreen && (
          <>
            <Col span={24}>
              <Row gutter={[16, 16]}>
                <Col>
                  <ViewControls
                    currentView={currentView}
                    isFullscreen={isFullscreen}
                    onViewChange={handleViewChange}
                    onFullscreenToggle={handleFullscreenToggle}
                  />
                  <Button
                    icon={<ScissorOutlined />}
                    onClick={handleCrop}
                    style={{ marginLeft: 8 }}
                    type="primary"
                  >
                    裁剪
                  </Button>
                </Col>
                <Col>
                  <ModeControls
                    xMode={xMode}
                    yMode={yMode}
                    zMode={zMode}
                    nbNeighbors={nbNeighbors}
                    stdRatio={stdRatio}
                    showPointCloud={showPointCloud}
                    onModeChange={(axis, value) => {
                      switch (axis) {
                        case 'x':
                          setXMode(value);
                          break;
                        case 'y':
                          setYMode(value);
                          break;
                        case 'z':
                          setZMode(value);
                          break;
                      }
                    }}
                    onNeighborsChange={setNbNeighbors}
                    onStdRatioChange={setStdRatio}
                    onShowPointCloudChange={setShowPointCloud}
                    onDenoise={handleDenoise}
                  />
                </Col>
              </Row>
            </Col>
          </>
        )}
        <Col span={24} style={{ position: 'relative' }}>
          <FilterCanvas
            imageRef={imageRef}
            isFullscreen={isFullscreen}
            drawMode={drawMode}
            xRegions={xRegions}
            yRegions={yRegions}
            lines={lines}
            onLineAdd={handleLineAdd}
          />
          {isFullscreen && (
            <CanvasOverlay
              drawMode={drawMode}
              isFullscreen={isFullscreen}
            >
              <ViewControls
                currentView={currentView}
                isFullscreen={isFullscreen}
                onViewChange={handleViewChange}
                onFullscreenToggle={handleFullscreenToggle}
              />
              <ModeControls
                xMode={xMode}
                yMode={yMode}
                zMode={zMode}
                nbNeighbors={nbNeighbors}
                stdRatio={stdRatio}
                showPointCloud={showPointCloud}
                onModeChange={(axis, value) => {
                  switch (axis) {
                    case 'x':
                      setXMode(value);
                      break;
                    case 'y':
                      setYMode(value);
                      break;
                    case 'z':
                      setZMode(value);
                      break;
                  }
                }}
                onNeighborsChange={setNbNeighbors}
                onStdRatioChange={setStdRatio}
                onShowPointCloudChange={setShowPointCloud}
                onDenoise={handleDenoise}
              />
              <Button
                icon={<ScissorOutlined />}
                onClick={handleCrop}
                type="primary"
              >
                裁剪
              </Button>
            </CanvasOverlay>
          )}
        </Col>
        {!isFullscreen && (
          <>
            <Col span={12}>
              <RegionList
                regions={xRegions}
                type="x"
                onDelete={(index) => handleRegionDelete(index, 'x')}
              />
            </Col>
            <Col span={12}>
              <RegionList
                regions={yRegions}
                type="y"
                onDelete={(index) => handleRegionDelete(index, 'y')}
              />
            </Col>
          </>
        )}
      </Row>
    </Card>
  );
};

export default FilterSection;
