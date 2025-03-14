import React, { useState, useEffect, useCallback } from 'react';
import { Card, Table, Statistic, Row, Col, message, Radio, Space, Switch } from 'antd';
import { Line, Column, Stock, Scatter } from '@ant-design/charts';
import { isNumber } from 'lodash';
import axios from '../utils/axios';

const ArcFittingComponent = () => {
  const [lineData, setLineData] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState(null);
  const [chartType, setChartType] = useState('line');
  const [showOutliers, setShowOutliers] = useState(false);
  const [statistics, setStatistics] = useState({
    overallMedian: 0,
    overallStd: 0,
    minValidRadius: 0,
    maxValidRadius: 0,
    totalValidCount: 0
  });

  // 列定义
  const columns = [
    {
      title: '线编号',
      dataIndex: 'lineIndex',
      key: 'lineIndex',
    },
    {
      title: '中位数半径',
      dataIndex: 'medianRadius',
      key: 'medianRadius',
      render: (value) => value.toFixed(3)
    },
    {
      title: '标准差',
      dataIndex: 'stdDev',
      key: 'stdDev',
      render: (value) => value.toFixed(6)
    },
    {
      title: '异常值统计',
      dataIndex: 'outlierStats',
      key: 'outlierStats',
      render: (stats) => `${stats.validCount}/${stats.totalCount} (有效/总数)`
    }
  ];

  const fetchSettings = async () => {
    try {
      const response = await axios.get('/api/settings/latest');
      if (response.data.status === 'success') {
        setSettings(response.data.data);
      } else {
        message.error('获取配置失败');
      }
    } catch (error) {
      console.error('获取配置失败:', error);
      message.error(error.response?.data?.error || '获取配置失败');
    }
  };

  const fetchData = useCallback(async () => {
    if (!settings) return;

    try {
      setLoading(true);
      const response = await axios.post('/api/arc-fitting-stats', settings.arcSettings);
      if (response.data.status === 'success') {
        setLineData(response.data.lineStats);
        setStatistics(response.data.overallStats);
      } else {
        message.error(response.data.error || '获取数据失败');
      }
    } catch (error) {
      console.error('获取圆拟合数据失败:', error);
      message.error(error.response?.data?.error || '请求失败');
    } finally {
      setLoading(false);
    }
  }, [settings]);

  // 组件加载时获取设置
  useEffect(() => {
    fetchSettings();
  }, []);

  // 设置更新后获取数据
  useEffect(() => {
    if (settings) {
      fetchData();
    }
  }, [settings, fetchData]);

  // 数据处理函数
  const processChartData = useCallback((data) => {
    let allRadiusData = [];
    data.forEach(line => {
      if (line.radiusData && Array.isArray(line.radiusData)) {
        const linePoints = line.radiusData
          .filter(point => showOutliers || point.isValid)
          .map(point => ({
            ...point,
            lineIndex: line.lineIndex + 1,
            type: point.isValid ? '有效值' : '异常值'
          }));
        allRadiusData = [...allRadiusData, ...linePoints];
      }
    });

    // 根据图表类型处理数据
    switch (chartType) {
      case 'line':
        return allRadiusData.map((point, idx) => ({
          index: idx + 1,
          radius: point.radius,
          type: point.type,
          lineIndex: point.lineIndex,
          iteration: point.iteration
        }));

      case 'stock':
        // 将数据分组，每10个点为一组
        const groupSize = 10;
        return Array.from({ length: Math.ceil(allRadiusData.length / groupSize) }, (_, i) => {
          const group = allRadiusData.slice(i * groupSize, (i + 1) * groupSize);
          const radii = group.map(p => p.radius);
          return {
            index: i + 1,
            low: Math.min(...radii),
            high: Math.max(...radii),
            open: radii[0],
            close: radii[radii.length - 1],
            count: group.length
          };
        });

      case 'scatter':
        return allRadiusData.map((point, idx) => ({
          index: idx + 1,
          radius: point.radius,
          type: point.type,
          lineIndex: point.lineIndex
        }));

      case 'column':
        // 计算半径值的分布
        const binSize = 0.1; // 柱状图的区间大小
        const bins = {};
        allRadiusData.forEach(point => {
          const bin = Math.floor(point.radius / binSize) * binSize;
          bins[bin] = (bins[bin] || 0) + 1;
        });
        return Object.entries(bins).map(([bin, count]) => ({
          radius: parseFloat(bin),
          count: count
        }));

      default:
        return [];
    }
  }, [chartType, showOutliers]);

  // 获取当前图表配置
  const getChartConfig = () => {
    const baseConfig = {
      animation: {
        appear: {
          animation: 'wave-in',
          duration: 1000
        }
      }
    };

    switch (chartType) {
      case 'line':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'index',
          yField: 'radius',
          seriesField: 'type',
          color: ['#2196F3', '#FF5252'],
          xAxis: { title: { text: '数据点序号' } },
          yAxis: { title: { text: '半径 (mm)' } },
          legend: { position: 'top-right' },
          tooltip: {
            showMarkers: true,
            formatter: (datum) => ({
              name: datum.type,
              value: `${datum.radius.toFixed(3)} mm\n行: ${datum.lineIndex}\n迭代: ${datum.iteration}`
            })
          },
          point: { size: 3, shape: 'circle' }
        };

      case 'stock':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'index',
          yField: ['open', 'close', 'high', 'low'],
          tooltip: {
            formatter: (datum) => ({
              name: '半径统计',
              value: 
                `最高: ${datum.high.toFixed(3)} mm\n` +
                `最低: ${datum.low.toFixed(3)} mm\n` +
                `起始: ${datum.open.toFixed(3)} mm\n` +
                `结束: ${datum.close.toFixed(3)} mm\n` +
                `数量: ${datum.count}`
            })
          }
        };

      case 'scatter':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'index',
          yField: 'radius',
          colorField: 'type',
          color: ['#2196F3', '#FF5252'],
          xAxis: { title: { text: '数据点序号' } },
          yAxis: { title: { text: '半径 (mm)' } },
          legend: { position: 'top-right' },
          tooltip: {
            formatter: (datum) => ({
              name: datum.type,
              value: `${datum.radius.toFixed(3)} mm\n行: ${datum.lineIndex}`
            })
          },
          shape: 'circle',
          size: 4
        };

      case 'column':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'radius',
          yField: 'count',
          xAxis: { 
            title: { text: '半径 (mm)' },
            label: { formatter: (v) => parseFloat(v).toFixed(2) }
          },
          yAxis: { title: { text: '数量' } },
          tooltip: {
            formatter: (datum) => ({
              name: '分布统计',
              value: `半径: ${datum.radius.toFixed(3)} mm\n数量: ${datum.count}`
            })
          },
          color: '#2196F3'
        };

      default:
        return baseConfig;
    }
  };

  useEffect(() => {
    if (lineData.length > 0) {
      const newChartData = processChartData(lineData);
      setChartData(newChartData);
    }
  }, [lineData, processChartData]);

  // 渲染当前选择的图表
  const renderChart = () => {
    const config = getChartConfig();
    switch (chartType) {
      case 'line':
        return <Line {...config} />;
      case 'stock':
        return <Stock {...config} />;
      case 'scatter':
        return <Scatter {...config} />;
      case 'column':
        return <Column {...config} />;
      default:
        return null;
    }
  };

  return (
    <Card title="圆弧拟合结果" bordered={false}>
      {/* 统计信息展示 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic 
            title="总体中位数" 
            value={statistics.overallMedian} 
            precision={3}
            suffix="mm"
          />
        </Col>
        <Col span={6}>
          <Statistic 
            title="总体标准差" 
            value={statistics.overallStd} 
            precision={6}
          />
        </Col>
        <Col span={6}>
          <Statistic 
            title="有效最小半径" 
            value={statistics.minValidRadius} 
            precision={3}
            suffix="mm"
          />
        </Col>
        <Col span={6}>
          <Statistic 
            title="有效最大半径" 
            value={statistics.maxValidRadius} 
            precision={3}
            suffix="mm"
          />
        </Col>
      </Row>

      {/* 图表控制 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Radio.Group value={chartType} onChange={e => setChartType(e.target.value)}>
            <Radio.Button value="line">折线图</Radio.Button>
            <Radio.Button value="stock">K线图</Radio.Button>
            <Radio.Button value="scatter">散点图</Radio.Button>
            <Radio.Button value="column">柱状图</Radio.Button>
          </Radio.Group>
          <Switch
            checkedChildren="显示异常值"
            unCheckedChildren="隐藏异常值"
            checked={showOutliers}
            onChange={setShowOutliers}
          />
        </Space>
      </div>

      {/* 图表展示 */}
      <div style={{ marginBottom: 24, height: 400 }}>
        {renderChart()}
      </div>

      {/* 数据表格 */}
      <Table
        columns={columns}
        dataSource={lineData}
        loading={loading}
        rowKey="lineIndex"
      />
    </Card>
  );
};

export default ArcFittingComponent;
