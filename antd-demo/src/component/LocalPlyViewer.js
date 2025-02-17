// ... (keep imports and constants)

// Status bar component
const StatusBar = ({ pointCount, hasColors, pointSize, format }) => (
  <div style={{ 
    padding: '8px 16px',
    background: '#f0f2f5',
    borderRadius: '4px',
    display: 'flex',
    gap: '16px',
    fontSize: '14px',
    color: '#666',
  }}>
    <div>点数量: {pointCount?.toLocaleString() || '无'}</div>
    <div>颜色信息: {hasColors ? '有' : '无'}</div>
    <div>点大小: {pointSize?.toFixed(4) || '未设置'}</div>
    <div>文件格式: {format || '未知'}</div>
  </div>
);

// Statistics component
const Statistics = ({ stats }) => (
  <div style={{ 
    position: 'absolute', 
    bottom: 16, 
    right: 16, 
    background: 'rgba(0, 0, 0, 0.6)',
    color: '#fff',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '12px',
    zIndex: 100,
  }}>
    <div>FPS: {stats.fps}</div>
    <div>面数: {stats.faces}</div>
    <div>渲染调用: {stats.calls}</div>
    <div>内存: {(stats.memory / 1024 / 1024).toFixed(2)} MB</div>
  </div>
);

// Main component
const LocalPlyViewer = () => {
  // ... (keep existing state declarations)
  const [stats, setStats] = useState({ fps: 0, faces: 0, calls: 0, memory: 0 });
  const statsRef = useRef({ frames: 0, lastTime: performance.now() });
  const [fileFormat, setFileFormat] = useState('');

  // Update stats
  useEffect(() => {
    const updateStats = () => {
      if (!renderer || !scene || !cloudData) return;

      const now = performance.now();
      const delta = now - statsRef.current.lastTime;

      if (delta > 1000) { // Update every second
        const fps = Math.round((statsRef.current.frames * 1000) / delta);
        
        setStats({
          fps,
          faces: scene.children.reduce((acc, obj) => acc + (obj.geometry?.attributes?.position?.count || 0), 0),
          calls: renderer.info.render.calls,
          memory: renderer.info.memory.geometries,
        });

        statsRef.current.frames = 0;
        statsRef.current.lastTime = now;
      }

      statsRef.current.frames++;
    };

    const animate = () => {
      updateStats();
      requestAnimationFrame(animate);
    };

    animate();
  }, [renderer, scene, cloudData]);

  // ... (keep existing methods)

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          本地 PLY 文件点云显示
          <Tooltip title={<ControlsHelp />} placement="bottom">
            <Button icon={<ControlOutlined />} type="text" size="small">
              操作说明
            </Button>
          </Tooltip>
        </div>
      }
      extra={
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button
            type="text"
            icon={showControls ? <EyeInvisibleOutlined /> : <EyeOutlined />}
            onClick={() => setShowControls(!showControls)}
          >
            {showControls ? '隐藏' : '显示'}控制面板
          </Button>
          <Tooltip title="按ESC退出全屏">
            <Button
              type="text"
              icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={() => {
                if (!document.fullscreenElement) {
                  canvasContainerRef.current?.requestFullscreen();
                  setIsFullscreen(true);
                } else {
                  document.exitFullscreen();
                  setIsFullscreen(false);
                }
              }}
            >
              全屏
            </Button>
          </Tooltip>
        </div>
      }
      style={{ width: '100%' }}
    >
      {showControls && (
        <>
          <div style={{ display: 'flex', gap: '8px', marginBottom: loading ? 8 : 16 }}>
            <Tooltip title="支持 ASCII 和二进制格式的 PLY 文件（最大500MB）">
              <Upload
                accept=".ply"
                showUploadList={false}
                beforeUpload={handleFileUpload}
              >
                <Button icon={<UploadOutlined />} loading={loading}>
                  {loading ? `处理中 ${progress}%` : '选择 PLY 文件'}
                </Button>
              </Upload>
            </Tooltip>

            <Tooltip title="重置视角以显示整个点云">
              <Button onClick={handleShowCloud} disabled={!hasCloud || loading}>
                显示点云
              </Button>
            </Tooltip>
            
            <Tooltip title="增大点的显示尺寸">
              <Button 
                icon={<PlusOutlined />}
                onClick={() => handlePointSizeChange(pointSize + POINT_SIZE_STEP)}
                disabled={!hasCloud}
              />
            </Tooltip>
            
            <Tooltip title="减小点的显示尺寸">
              <Button 
                icon={<MinusOutlined />}
                onClick={() => handlePointSizeChange(pointSize - POINT_SIZE_STEP)}
                disabled={!hasCloud}
              />
            </Tooltip>
            
            <Tooltip title="切换背景颜色">
              <Button 
                icon={<BgColorsOutlined />}
                onClick={() => {
                  if (renderer) {
                    const newState = !isDarkBackground;
                    setIsDarkBackground(newState);
                    renderer.setClearColor(newState ? 0x000000 : 0xffffff, 1);
                    updateGridColors(newState);
                  }
                }}
              />
            </Tooltip>
            
            <div style={{ marginLeft: 'auto' }}>
              <Tooltip title="显示/隐藏参考线">
                <Button
                  onClick={() => {
                    const { axis, grid } = helpersRef.current;
                    const newState = !showHelpers;
                    setShowHelpers(newState);
                    axis.visible = newState;
                    grid.visible = newState;
                  }}
                >
                  {showHelpers ? '隐藏' : '显示'}参考线
                </Button>
              </Tooltip>
            </div>
          </div>
          
          <PointCloudControls
            pointSize={pointSize}
            setPointSize={handlePointSizeChange}
            isEnabled={hasCloud && !loading}
          />

          {hasCloud && (
            <StatusBar 
              pointCount={cloudData?.geometry?.attributes?.position?.count}
              hasColors={!!cloudData?.geometry?.attributes?.color}
              pointSize={pointSize}
              format={fileFormat}
            />
          )}
        </>
      )}
      
      {loading && (
        <div style={{ marginBottom: 16 }}>
          <Progress 
            percent={progress} 
            status="active" 
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
          />
        </div>
      )}
      
      <div
        ref={canvasContainerRef}
        style={{
          width: isFullscreen ? '100vw' : '100%',
          height: isFullscreen ? '100vh' : '600px',
          position: 'relative',
          outline: 'none',
          tabIndex: -1,
          cursor: 'grab',
        }}
        onMouseDown={() => {
          if (canvasContainerRef.current) {
            canvasContainerRef.current.style.cursor = 'grabbing';
          }
        }}
        onMouseUp={() => {
          if (canvasContainerRef.current) {
            canvasContainerRef.current.style.cursor = 'grab';
          }
        }}
        onKeyDown={handleKeyDown}
      >
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
        <Statistics stats={stats} />
      </div>
    </Card>
  );
};

export default LocalPlyViewer;
