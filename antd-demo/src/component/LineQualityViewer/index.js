import React, { useState, useCallback, useEffect } from 'react';
import { Card, Space, message, Button, Input, Divider, Alert } from 'antd';
// import axios from 'axios';
import axios from '../../utils/axios';
import AxisSelector from './AxisSelector';
import LineChart from './LineChart';
import ExportPanel from './ExportPanel';
import ControlPanel from './ControlPanel';
import ModelPanel from './ModelPanel';

const LineQualityViewer = () => {
  const [selectedAxis, setSelectedAxis] = useState('x');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [defectLines, setDefectLines] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [total, setTotal] = useState(0);
  const [autoDetectEnabled, setAutoDetectEnabled] = useState(false);
  const [currentProbability, setCurrentProbability] = useState(null);

  // 获取分组数据（不使用缓存）
  const fetchGroupedData = useCallback(async (axis, index) => {
    try {
      setLoading(true);
      const response = await axios.post('/api/point-cloud/group-points', { 
        axis,
        index
      });

      setData(response.data);
      setTotal(response.data.total_groups);
    } catch (error) {
      console.error('获取分组数据失败:', error);
      message.error('获取数据失败');
    } finally {
      setLoading(false);
    }
  }, []);

  // 初始加载
  useEffect(() => {
    fetchGroupedData(selectedAxis, 0);
  }, [selectedAxis, fetchGroupedData]);

  // 自动检测
  const handleDetect = async (points) => {
    try {
      setLoading(true);
      const response = await axios.post('/api/model/predict', {
        points: points.map(point => point[2])
      });
      setCurrentProbability(response.data.probability);
    } catch (error) {
      console.error('检测失败:', error);
      message.error('检测失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (autoDetectEnabled && data?.group?.points) {
      handleDetect(data.group.points);
    } else if (!autoDetectEnabled) {
      setCurrentProbability(null);
    }
  }, [currentIndex, data, autoDetectEnabled]);

  // 线条切换
  const handlePrev = useCallback(() => {
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1;
      setCurrentIndex(newIndex);
      fetchGroupedData(selectedAxis, newIndex);
    }
  }, [currentIndex, selectedAxis, fetchGroupedData]);

  const handleNext = useCallback(() => {
    if (currentIndex < total - 1) {
      const newIndex = currentIndex + 1;
      setCurrentIndex(newIndex);
      fetchGroupedData(selectedAxis, newIndex);
    }
  }, [currentIndex, total, selectedAxis, fetchGroupedData]);

  const handleJump = useCallback((index) => {
    const newIndex = Math.max(0, Math.min(index, total - 1));
    setCurrentIndex(newIndex);
    fetchGroupedData(selectedAxis, newIndex);
  }, [total, selectedAxis, fetchGroupedData]);

  // 轴切换
  const handleAxisChange = useCallback((value) => {
    setSelectedAxis(value);
    setCurrentIndex(0);
    fetchGroupedData(value, 0);
  }, [fetchGroupedData]);

  // 标记缺陷
  const handleMarkDefect = useCallback(() => {
    setDefectLines(prev => {
      const newDefectLines = [...prev];
      newDefectLines[currentIndex] = !newDefectLines[currentIndex];
      return newDefectLines;
    });
  }, [currentIndex]);

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.target.tagName === 'INPUT') return;

      switch (event.key.toLowerCase()) {
        case 'arrowleft':
          handlePrev();
          break;
        case 'arrowright':
          handleNext();
          break;
        case 'q':
          handleMarkDefect();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePrev, handleNext, handleMarkDefect]);

  return (
    <Card title="线质量分析" bordered={false}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card size="small" title="选择分组轴">
          <AxisSelector 
            selectedAxis={selectedAxis}
            onAxisChange={handleAxisChange}
          />
        </Card>

        <Card size="small" title="线条设置">
          <Space direction="vertical" style={{ width: '100%' }} size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <ControlPanel 
                  onMarkDefect={handleMarkDefect}
                  isDefect={defectLines[currentIndex] || false}
                  loading={loading}
                />
                <ModelPanel 
                  autoDetectEnabled={autoDetectEnabled}
                  onToggleAutoDetect={setAutoDetectEnabled}
                />
              </Space>
              {currentProbability !== null && (
                <Alert
                  message={`当前线条缺陷置信度: ${(currentProbability * 100).toFixed(2)}%`}
                  type="info"
                  showIcon
                />
              )}
            </Space>
            <Divider style={{ margin: '12px 0' }} />
            <Space style={{ width: '100%', justifyContent: 'center' }}>
              <Button 
                onClick={handlePrev} 
                disabled={currentIndex <= 0}
              >
                上一条
              </Button>
              <span style={{ margin: '0 16px' }}>
                {total > 0 ? `${currentIndex + 1} / ${total}` : '暂无数据'}
              </span>
              <Button 
                onClick={handleNext}
                disabled={currentIndex >= total - 1}
              >
                下一条
              </Button>
              <Input
                style={{ width: 80 }}
                type="number"
                min={1}
                max={total}
                value={currentIndex + 1}
                onChange={e => {
                  const val = parseInt(e.target.value, 10);
                  if (!isNaN(val)) {
                    handleJump(val - 1);
                  }
                }}
              />
            </Space>
          </Space>
        </Card>

        <Card size="small" title="线质量可视化">
          <center>
            <LineChart 
              key={`${selectedAxis}_${currentIndex}`} 
              data={data}
              loading={loading}
              selectedAxis={selectedAxis}
            />
          </center>
        </Card>

        <Card size="small" title="数据导出">
          <Space>
            <ExportPanel 
              data={data}
              defectLines={defectLines}
              loading={loading}
              total={total}
            />
          </Space>
        </Card>
      </Space>
    </Card>
  );
};

export default LineQualityViewer;
