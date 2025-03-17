import React, { useEffect, useRef, useState } from 'react';
import { Card, Button, Row, Col, List, Typography, message, Select, Input, InputNumber, Checkbox } from 'antd';
import { FullscreenOutlined, FullscreenExitOutlined, DeleteOutlined, ScissorOutlined, CloudOutlined, EyeOutlined } from '@ant-design/icons';
import axios from '../utils/axios';
const { Text } = Typography;

const FilterSection = () => {
  const canvasRef = useRef(null);
  const imageRef = useRef(new Image());
  const [currentView, setCurrentView] = useState('xy');
  const [lines, setLines] = useState([]);
  const [xRegions, setXRegions] = useState([]);
  const [yRegions, setYRegions] = useState([]);
  const [coordinateRanges, setCoordinateRanges] = useState({});
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [drawMode, setDrawMode] = useState('x'); // 'x' for vertical lines, 'y' for horizontal
  const [xMode, setXMode] = useState('keep'); // 'keep' or 'remove'
  const [yMode, setYMode] = useState('keep'); // 'keep' or 'remove'
  const [zMode, setZMode] = useState('keep'); // 'keep' or 'remove'
  const [nbNeighbors, setNbNeighbors] = useState(100);
  const [stdRatio, setStdRatio] = useState(0.5);
  const [showPointCloud, setShowPointCloud] = useState(false);

  // Add new styles for fullscreen controls
  const fullscreenControlsStyle = {
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
  };

  const fullscreenInfoStyle = {
    position: 'fixed',
    bottom: '20px',
    left: '20px',
    zIndex: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: '10px',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
  };

  const fullscreenListStyle = {
    position: 'fixed',
    left: '20px',
    top: '20px',
    zIndex: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: '10px',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
    maxHeight: 'calc(100vh - 100px)',
    overflowY: 'auto',
    width: '250px'
  };

  // Render fullscreen controls
  const FullscreenControls = () => (
    <>
      <div style={fullscreenListStyle}>
        <Row gutter={[0, 16]}>
          <Col span={24}>
            <List
              size="small"
              header={<Text strong>垂直区域 (X):</Text>}
              dataSource={xRegions.map((region, index) => ({
                region,
                index
              }))}
              renderItem={({ region, index }) => (
                <List.Item
                  actions={[
                    <Button 
                      type="link" 
                      danger 
                      icon={<DeleteOutlined />}
                      onClick={() => handleDeleteRegion(index, 'x')}
                    >
                      删除
                    </Button>
                  ]}
                >
                  区域 {index + 1}: [{region[0].toFixed(2)}, {region[1].toFixed(2)}]
                </List.Item>
              )}
            />
          </Col>
          <Col span={24}>
            <List
              size="small"
              header={<Text strong>水平区域 (Y):</Text>}
              dataSource={yRegions.map((region, index) => ({
                region,
                index
              }))}
              renderItem={({ region, index }) => (
                <List.Item
                  actions={[
                    <Button 
                      type="link" 
                      danger 
                      icon={<DeleteOutlined />}
                      onClick={() => handleDeleteRegion(index, 'y')}
                    >
                      删除
                    </Button>
                  ]}
                >
                  区域 {index + 1}: [{region[0].toFixed(2)}, {region[1].toFixed(2)}]
                </List.Item>
              )}
            />
          </Col>
        </Row>
      </div>
      <div style={fullscreenControlsStyle}>
        <Row gutter={[16, 16]}>
          <Col>
            <Button.Group>
              <Button onClick={() => setCurrentView('xy')}>XY</Button>
              <Button onClick={() => setCurrentView('xz')}>XZ</Button>
              <Button onClick={() => setCurrentView('yz')}>YZ</Button>
            </Button.Group>
          </Col>
          <Col>
            <Select
              value={xMode}
              onChange={setXMode}
              style={{ width: 120, marginRight: 8 }}
              options={[
                { value: 'keep', label: 'X: 保留' },
                { value: 'remove', label: 'X: 移除' },
              ]}
            />
            <Select
              value={yMode}
              onChange={setYMode}
              style={{ width: 120, marginRight: 8 }}
              options={[
                { value: 'keep', label: 'Y: 保留' },
                { value: 'remove', label: 'Y: 移除' },
              ]}
            />
            <Select
              value={zMode}
              onChange={setZMode}
              style={{ width: 120, marginRight: 8 }}
              options={[
                { value: 'keep', label: 'Z: 保留' },
                { value: 'remove', label: 'Z: 移除' },
              ]}
            />
            <InputNumber
              value={nbNeighbors}
              onChange={setNbNeighbors}
              style={{ width: 120, marginRight: 4 }}
              min={1}
              max={1000}
              addonBefore="邻居点数"
            />
            <InputNumber
              value={stdRatio}
              onChange={setStdRatio}
              style={{ width: 120, marginRight: 8 }}
              min={0.1}
              max={2}
              step={0.1}
              addonBefore="标准差比例"
            />
            <Button
              type="primary"
              danger
              icon={<CloudOutlined />}
              onClick={handleDenoise}
              style={{ marginRight: 8 }}
            >
              去噪
            </Button>
            <Checkbox
              checked={showPointCloud}
              onChange={(e) => setShowPointCloud(e.target.checked)}
              style={{ marginLeft: 8 }}
            >
              <EyeOutlined /> 显示点云
            </Checkbox>
          </Col>
        </Row>
        <Button 
          icon={<FullscreenExitOutlined />}
          onClick={toggleFullscreen}
        >
          退出全屏
        </Button>
        <Button
          icon={<ScissorOutlined />}
          onClick={handleCrop}
          type="primary"
        >
          裁剪
        </Button>
      </div>
      <div style={fullscreenInfoStyle}>
        <Text type="secondary">
          当前模式: {drawMode === 'x' ? '垂直选择' : '水平选择'} | 
          按键说明: M切换模式，Enter确认区域，R撤销上一步
        </Text>
      </div>
    </>
  );

  const handleResize = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // 设置固定的画布大小
    const containerWidth = isFullscreen ? window.innerWidth * 0.8 : canvas.parentElement.clientWidth;
    const containerHeight = isFullscreen ? window.innerHeight * 0.8 : window.innerHeight * 0.7;
    
    const imageAspectRatio = imageRef.current.width / imageRef.current.height;
    const containerAspectRatio = containerWidth / containerHeight;
    
    let width, height;
    if (imageAspectRatio > containerAspectRatio) {
      width = containerWidth;
      height = containerWidth / imageAspectRatio;
    } else {
      height = containerHeight;
      width = containerHeight * imageAspectRatio;
    }
    
    // Set display size
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    
    // Set actual size to match display size
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    
    // Scale context to ensure correct drawing operations
    const ctx = canvas.getContext('2d');
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    if (isFullscreen) {
      canvas.style.position = 'fixed';
      canvas.style.left = `${(window.innerWidth - width) / 2}px`;
      canvas.style.top = `${(window.innerHeight - height) / 2}px`;
    } else {
      canvas.style.position = 'static';
      canvas.style.left = 'auto';
      canvas.style.top = 'auto';
    }
    
    redrawCanvas();
  };

  // Every time the view changes, load image and reset regions
  const loadImage = async (view) => {
    try {
      const response = await axios.get(`/api/files/img/${view}?t=${Date.now()}`, {
        responseType: 'blob'
      });
      const blob = new Blob([response.data], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);
      
      // 清理旧的 URL 对象
      if (imageRef.current.src && imageRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(imageRef.current.src);
      }
      
      imageRef.current.src = imageUrl;
      imageRef.current.onload = () => {
        handleResize();
      };
    } catch (error) {
      console.error('加载图像失败:', error);
      message.error('加载图像失败，请检查认证状态');
    }
  };

  useEffect(() => {
    console.log(`Loading new view: ${currentView}`);
    loadImage(currentView);
    fetchCoordinateRanges(currentView);
    setXRegions([]);
    setYRegions([]);
    setLines([]);
  }, [currentView]);

  // Component mount initialization
  useEffect(() => {
    console.log('Component mounted, initializing currentView:', currentView);
    loadImage(currentView);
    fetchCoordinateRanges(currentView);
    setXRegions([]);
    setYRegions([]);
    setLines([]);
  }, []);

  // Update resize effect to handle fullscreen changes
  useEffect(() => {
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, [isFullscreen]);

  // Monitor fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      const fullscreenElement = document.fullscreenElement;
      setIsFullscreen(!!fullscreenElement);
      
      if (fullscreenElement) {
        fullscreenElement.style.backgroundColor = '#f0f2f5';
        handleResize();
      } else {
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.style.position = 'static';
          canvas.style.left = 'auto';
          canvas.style.top = 'auto';
          handleResize();
        }
      }
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Toggle fullscreen with improved handling
  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        const canvas = canvasRef.current;
        await canvas.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error('Fullscreen error:', err);
      message.error('全屏模式切换失败');
    }
  };

  const handleDeleteRegion = (index, type) => {
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
  };

  const handleDenoise = async () => {
    try {
      await axios.post('/denoise', {
        nb_neighbors: nbNeighbors,
        std_ratio: stdRatio,
        settings: {
          show: showPointCloud
        }
      });

      message.success('去噪处理成功');
      loadImage(currentView);
      setLines([]);
      setXRegions([]);
      setYRegions([]);
      // const timestamp = new Date().getTime();
      // imageRef.current.src = `/api/files/img/${currentView}?t=${timestamp}`;
    } catch (error) {
      console.error('Denoise error:', error);
      message.error('去噪处理失败');
    }
  };

  // Handle crop operation
  const handleCrop = async () => {
    if (xRegions.length === 0 && yRegions.length === 0) {
      message.warning('请先选择裁剪区域');
      return;
    }

    // Normalize regions using coordinate ranges
    const normalizeRegions = (regions, axis, isVertical = false) => {
      const min = coordinateRanges[`${axis}_min`];
      const max = coordinateRanges[`${axis}_max`];
      
      return regions.map(([start, end]) => {
        // 直接使用比例值转换到实际坐标
        return [
          min + (max - min) * start,
          min + (max - min) * end
        ];
      });
    };

    // Transform regions based on the current view and normalize
    let regions;
    switch (currentView) {
      case 'xy':
        regions = {
          x_regions: normalizeRegions(xRegions, 'x'),
          y_regions: normalizeRegions(yRegions, 'y', true),
          z_regions: []
        };
        break;
      case 'xz':
        regions = {
          x_regions: normalizeRegions(xRegions, 'x'),
          y_regions: [],
          z_regions: normalizeRegions(yRegions, 'z', true)
        };
        break;
      case 'yz':
        regions = {
          x_regions: [],
          y_regions: normalizeRegions(xRegions, 'y'),
          z_regions: normalizeRegions(yRegions, 'z', true)
        };
        break;
      default:
        regions = {
          x_regions: [],
          y_regions: [],
          z_regions: []
        };
    }

    try {
      console.log('Sending crop request:', JSON.stringify({
        regions,
        modes: {
          x_mode: xMode,
          y_mode: yMode,
          z_mode: zMode
        }
      }));
      
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
      }
    );

      setXRegions([]);
      setYRegions([]);
      setLines([]);
      message.success('裁剪成功');
      
      loadImage(currentView);
      // const timestamp = new Date().getTime();
      // imageRef.current.src = `/api/files/img/${currentView}?t=${timestamp}`;
    } catch (error) {
      console.error('Crop error:', error);
      message.error('裁剪操作失败');
    }
  };

  const fetchCoordinateRanges = async (view) => {
    try {
      const response = await axios.get(`/api/files/yml/info?t=${Date.now()}`);
      const ranges = parseYaml(response.data);
      console.log('Fetching YAML:', ranges);
      setCoordinateRanges(ranges);
    } catch (error) {
      console.error('Error fetching YAML:', error);
    }
  };

  const parseYaml = (yamlText) => {
    const lines = yamlText.split('\n').filter((line) => line.includes(':'));
    const result = {};
    lines.forEach((line) => {
      const [key, value] = line.split(':').map((str) => str.trim());
      result[key] = parseFloat(value);
    });
    return result;
  };

  // Redraw canvas with image and selected region lines
  const redrawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width / window.devicePixelRatio;
    const height = canvas.height / window.devicePixelRatio;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imageRef.current, 0, 0, width, height);

    // Draw regions with fill using relative positions
    xRegions.forEach((region) => {
      const [x1, x2] = region;
      const pixelX1 = x1 * width;
      const pixelX2 = x2 * width;
      
      ctx.fillStyle = 'rgba(0, 0, 255, 0.1)';
      ctx.fillRect(pixelX1, 0, pixelX2 - pixelX1, height);
      
      ctx.strokeStyle = 'blue';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pixelX1, 0);
      ctx.lineTo(pixelX1, height);
      ctx.moveTo(pixelX2, 0);
      ctx.lineTo(pixelX2, height);
      ctx.stroke();
    });

    yRegions.forEach((region) => {
      const [y1, y2] = region;
      const pixelY1 = y1 * height;
      const pixelY2 = y2 * height;
      
      ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
      ctx.fillRect(0, pixelY1, width, pixelY2 - pixelY1);
      
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, pixelY1);
      ctx.lineTo(width, pixelY1);
      ctx.moveTo(0, pixelY2);
      ctx.lineTo(width, pixelY2);
      ctx.stroke();
    });

    // Draw current lines using relative positions
    lines.forEach(line => {
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (drawMode === 'x') {
        const pixelX = line.x * width;
        ctx.moveTo(pixelX, 0);
        ctx.lineTo(pixelX, height);
      } else {
        const pixelY = line.y * height;
        ctx.moveTo(0, pixelY);
        ctx.lineTo(width, pixelY);
      }
      ctx.stroke();
    });
  };

  // Listen for canvas mouse events for drawing region lines
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseUp = (e) => {
      const rect = canvas.getBoundingClientRect();
      
      // Convert click position to relative coordinates (0-1)
      const relativeX = (e.clientX - rect.left) / rect.width;
      const relativeY = (e.clientY - rect.top) / rect.height;
      
      // Store relative position
      const newLine = drawMode === 'x' ? { x: relativeX } : { y: relativeY };
      setLines(prev => [...prev, newLine]);
    };

    canvas.addEventListener('mouseup', handleMouseUp);
    return () => canvas.removeEventListener('mouseup', handleMouseUp);
  }, [drawMode]);

  // Listen for keyboard events
  useEffect(() => {
    const handleKeyDown = async (e) => {
      // Mode switching with X/Y/Z keys
      if (e.key.toLowerCase() === 'x') {
        e.preventDefault();
        setXMode(prev => prev === 'keep' ? 'remove' : 'keep');
        return;
      }
      if (e.key.toLowerCase() === 'y') {
        e.preventDefault();
        setYMode(prev => prev === 'keep' ? 'remove' : 'keep');
        return;
      }
      if (e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setZMode(prev => prev === 'keep' ? 'remove' : 'keep');
        return;
      }

      // Region and drawing controls
      if (e.key === 'Enter' && lines.length >= 2) {
        e.preventDefault();
        if (drawMode === 'x') {
          const xCoords = lines.map(line => line.x).sort((a, b) => a - b);
          const start = xCoords[0];
          const end = xCoords[xCoords.length - 1];
          if (start !== undefined && end !== undefined) {
            await setXRegions(prev => {
              console.log('Adding X region:', [start, end]);
              return [...prev, [start, end]];
            });
          }
        } else {
          const yCoords = lines.map(line => line.y).sort((a, b) => a - b);
          const start = yCoords[0];
          const end = yCoords[yCoords.length - 1];
          if (start !== undefined && end !== undefined) {
            await setYRegions(prev => {
              console.log('Adding Y region:', [start, end]);
              return [...prev, [start, end]];
            });
          }
        }
        setLines([]);
        requestAnimationFrame(redrawCanvas);
      }
      
      if (e.key.toLowerCase() === 'r' && lines.length > 0) {
        e.preventDefault();
        setLines(prev => prev.slice(0, -1));
      }
      
      if (e.key.toLowerCase() === 'm') {
        e.preventDefault();
        setDrawMode(prev => prev === 'x' ? 'y' : 'x');
        setLines([]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [lines, drawMode, setXMode, setYMode, setZMode]);

  // Redraw when any state changes
  useEffect(() => {
    redrawCanvas();
  }, [lines, xRegions, yRegions, drawMode, xMode, yMode, zMode]);

  return (
    <Card bordered={false}>
      <Row gutter={[16, 16]}>
        {!isFullscreen && (
          <>
            <Col span={24}>
              <Row gutter={[16, 16]}>
                <Col>
                  <Button.Group>
                    <Button onClick={() => setCurrentView('xy')}>XY</Button>
                    <Button onClick={() => setCurrentView('xz')}>XZ</Button>
                    <Button onClick={() => setCurrentView('yz')}>YZ</Button>
                  </Button.Group>
                  <Button 
                    icon={<FullscreenOutlined />}
                    onClick={toggleFullscreen}
                    style={{ marginLeft: 8 }}
                  >
                    全屏
                  </Button>
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
                  <Select
                    value={xMode}
                    onChange={setXMode}
                    style={{ width: 95, marginRight: 8 }}
                    options={[
                      { value: 'keep', label: 'X: 保留' },
                      { value: 'remove', label: 'X: 移除' },
                    ]}
                  />
                  <Select
                    value={yMode}
                    onChange={setYMode}
                    style={{ width: 95, marginRight: 8 }}
                    options={[
                      { value: 'keep', label: 'Y: 保留' },
                      { value: 'remove', label: 'Y: 移除' },
                    ]}
                  />
                  <Select
                    value={zMode}
                    onChange={setZMode}
                    style={{ width: 95, marginRight: 8 }}
                    options={[
                      { value: 'keep', label: 'Z: 保留' },
                      { value: 'remove', label: 'Z: 移除' },
                    ]}
                  />
                  <InputNumber
                    value={nbNeighbors}
                    onChange={setNbNeighbors}
                    style={{ width: 120, marginRight: 4 }}
                    min={1}
                    max={1000}
                    addonBefore="nb数"
                  />
                  <InputNumber
                    value={stdRatio}
                    onChange={setStdRatio}
                    style={{ width: 120, marginRight: 8 }}
                    min={0.1}
                    max={2}
                    step={0.1}
                    addonBefore="std比例"
                  />
                  <Button
                    type="primary"
                    danger
                    icon={<CloudOutlined />}
                    onClick={handleDenoise}
                    style={{ marginRight: 8 }}
                  >
                    去噪
                  </Button>
                  <Checkbox
                    checked={showPointCloud}
                    onChange={(e) => setShowPointCloud(e.target.checked)}
                    style={{ marginLeft: 8 }}
                  >
                    <EyeOutlined /> 显示点云
                  </Checkbox>
                </Col>
              </Row>
            </Col>
            <Col span={24}>
              <Text type="secondary">
                当前模式: {drawMode === 'x' ? '垂直选择' : '水平选择'} | 
                按键说明: M切换模式，Enter确认区域，R撤销上一步，XYZ切换保留/移除，C快速裁减
              </Text>
            </Col>
          </>
        )}
        <Col span={24} style={{ position: 'relative' }}>
          <canvas 
            ref={canvasRef} 
            style={{ 
              border: '1px solid #d9d9d9',
              width: '100%',
              cursor: 'crosshair',
              transition: 'all 0.3s ease'
            }} 
          />
          {isFullscreen && <FullscreenControls />}
        </Col>
        {!isFullscreen && (
          <>
            <Col span={12}>
              <List
                header={<Text strong>垂直区域 (X):</Text>}
                dataSource={xRegions.map((region, index) => ({
                  region,
                  index
                }))}
                renderItem={({ region, index }) => (
                  <List.Item
                    actions={[
                      <Button 
                        type="link" 
                        danger 
                        icon={<DeleteOutlined />}
                        onClick={() => handleDeleteRegion(index, 'x')}
                      >
                        删除
                      </Button>
                    ]}
                  >
                    区域 {index + 1}: [{region[0].toFixed(2)}, {region[1].toFixed(2)}]
                  </List.Item>
                )}
              />
            </Col>
            <Col span={12}>
              <List
                header={<Text strong>水平区域 (Y):</Text>}
                dataSource={yRegions.map((region, index) => ({
                  region,
                  index
                }))}
                renderItem={({ region, index }) => (
                  <List.Item
                    actions={[
                      <Button 
                        type="link" 
                        danger 
                        icon={<DeleteOutlined />}
                        onClick={() => handleDeleteRegion(index, 'y')}
                      >
                        删除
                      </Button>
                    ]}
                  >
                    区域 {index + 1}: [{region[0].toFixed(2)}, {region[1].toFixed(2)}]
                  </List.Item>
                )}
              />
            </Col>
          </>
        )}
      </Row>
    </Card>
  );
};

export default FilterSection;
