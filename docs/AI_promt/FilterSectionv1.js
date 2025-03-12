import React, { useEffect, useRef, useState } from 'react';
import { Card, Button, Row, Col, List, Typography, message, Select } from 'antd';
import { FullscreenOutlined, FullscreenExitOutlined, DeleteOutlined, ScissorOutlined } from '@ant-design/icons';
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
              style={{ width: 120 }}
              options={[
                { value: 'keep', label: 'Z: 保留' },
                { value: 'remove', label: 'Z: 移除' },
              ]}
            />
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
    
    const containerWidth = isFullscreen ? window.innerWidth : canvas.parentElement.clientWidth;
    const containerHeight = isFullscreen ? window.innerHeight : window.innerHeight * 0.6;
    
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

  // Every time the view changes, load image and reset regions 每次视图更改时，都会加载图像并重置区域 
  useEffect(() => {
    console.log('Loading new view:', currentView);
    imageRef.current.src = `http://localhost:9304/img/${currentView}`;
    imageRef.current.onload = () => {
      handleResize();
    };
    fetchCoordinateRanges(currentView);
    setXRegions([]);
    setYRegions([]);
    setLines([]);
  }, [currentView]);

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

  // Handle region deletion
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
      const canvas = canvasRef.current;
      
      // Calculate scale factors
      const scaleX = canvas.width / window.devicePixelRatio / imageRef.current.width;
      const scaleY = canvas.height / window.devicePixelRatio / imageRef.current.height;
      
      return regions.map(([start, end]) => {
        // Convert from screen coordinates to image coordinates
        const startPx = isVertical ? start / scaleY : start / scaleX;
        const endPx = isVertical ? end / scaleY : end / scaleX;
        
        // Convert from image coordinates to real-world coordinates
        console.log('Converting:', coordinateRanges,startPx, endPx, min, max);
        return [
          min + (max - min) * (startPx / (isVertical ? imageRef.current.height : imageRef.current.width)),
          min + (max - min) * (endPx / (isVertical ? imageRef.current.height : imageRef.current.width))
        ];
      });
    };

    // Transform regions based on the current view and normalize
    // 根据当前视图转换区域并进行归一化
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

    console.log('Screen dimensions:', {
      width: canvasRef.current.width / window.devicePixelRatio,
      height: canvasRef.current.height / window.devicePixelRatio
    });
    console.log('Image dimensions:', {
      width: imageRef.current.width,
      height: imageRef.current.height
    });
    console.log('Current coordinate ranges:', coordinateRanges);
    console.log('Sending crop request with normalized regions:', regions);

    try {
      console.log('Sending crop request:', JSON.stringify({
        regions,
        modes: {
          x_mode: xMode,
          y_mode: yMode,
          z_mode: zMode
        }
      }));
      const response = await fetch('http://localhost:9304/crop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          regions,
          modes: {
            x_mode: xMode,
            y_mode: yMode,
            z_mode: zMode
          }
        })
      });

      if (response.ok) {
        message.success('裁剪成功');
        // Refresh image with cache busting
        const timestamp = new Date().getTime();
        imageRef.current.src = `http://localhost:9304/img/${currentView}?t=${timestamp}`;
      } else {
        const errorData = await response.json();
        message.error(`裁剪失败: ${errorData.error || '未知错误'}`);
      }
    } catch (error) {
      console.error('Crop error:', error);
      message.error('裁剪操作失败');
    }
  };

  // Get YAML coordinate ranges and parse
  const fetchCoordinateRanges = async (view) => {
    try {
      const response = await fetch(`http://localhost:9304/yml/info`);
      if (!response.ok) return;
      const text = await response.text();
      const ranges = parseYaml(text);
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
    ctx.clearRect(0, 0, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio);

    // Draw regions with fill
    xRegions.forEach((region) => {
      const [x1, x2] = region;
      ctx.fillStyle = 'rgba(0, 0, 255, 0.1)';
      ctx.fillRect(x1, 0, x2 - x1, canvas.height / window.devicePixelRatio);
      
      ctx.strokeStyle = 'blue';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, 0);
      ctx.lineTo(x1, canvas.height / window.devicePixelRatio);
      ctx.moveTo(x2, 0);
      ctx.lineTo(x2, canvas.height / window.devicePixelRatio);
      ctx.stroke();
    });

    yRegions.forEach((region) => {
      const [y1, y2] = region;
      ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
      ctx.fillRect(0, y1, canvas.width / window.devicePixelRatio, y2 - y1);
      
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, y1);
      ctx.lineTo(canvas.width / window.devicePixelRatio, y1);
      ctx.moveTo(0, y2);
      ctx.lineTo(canvas.width / window.devicePixelRatio, y2);
      ctx.stroke();
    });

    // Draw current lines
    lines.forEach(line => {
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (drawMode === 'x') {
        ctx.moveTo(line.x, 0);
        ctx.lineTo(line.x, canvas.height / window.devicePixelRatio);
      } else {
        ctx.moveTo(0, line.y);
        ctx.lineTo(canvas.width / window.devicePixelRatio, line.y);
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
      const scaleX = (canvas.width / window.devicePixelRatio) / rect.width;
      const scaleY = (canvas.height / window.devicePixelRatio) / rect.height;
      
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      
      const newLine = drawMode === 'x' ? { x } : { y };
      setLines(prev => [...prev, newLine]);
    };

    canvas.addEventListener('mouseup', handleMouseUp);
    return () => canvas.removeEventListener('mouseup', handleMouseUp);
  }, [drawMode]);

  // Listen for keyboard events: Enter to confirm region, R to undo last action, M to switch mode
  useEffect(() => {
    const handleKeyDown = async (e) => {
      if (e.key === 'Enter' && lines.length >= 2) {
        e.preventDefault(); // Prevent form submission
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
  }, [lines, drawMode]);

  // Redraw when any state changes
  useEffect(() => {
    redrawCanvas();
  }, [lines, xRegions, yRegions, drawMode]);

  return (
    <Card title="点云预处理" bordered={false}>
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
                    style={{ width: 120 }}
                    options={[
                      { value: 'keep', label: 'Z: 保留' },
                      { value: 'remove', label: 'Z: 移除' },
                    ]}
                  />
                </Col>
              </Row>
            </Col>
            <Col span={24}>
              <Text type="secondary">
                当前模式: {drawMode === 'x' ? '垂直选择' : '水平选择'} | 
                按键说明: M切换模式，Enter确认区域，R撤销上一步
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
