import React, { useState, useEffect, useCallback } from 'react';
import { Card, Table, Statistic, Row, Col, message, Radio, Space, Switch, Button, Tooltip } from 'antd';
import { DownloadOutlined, CopyOutlined } from '@ant-design/icons';
import { Line, Column, Stock, Scatter } from '@ant-design/charts';
import { isNumber } from 'lodash';
import axios from '../utils/axios';
const defaultArcSettings = {
  arcMethod: 'HyperFit',
  arcNormalNeighbors: 20,
  arcMaxRadius: 12,
  arcMinRadius: 6,
  learningRate: 0.01,
  gradientMaxIterations: 1000,
  tolerance: 1e-6,
  fitIterations: 50,      // 拟合迭代次数n
  samplePercentage: 50    // 采样百分比m
};

const defaultCylinderSettings = {
  cylinderMethod: 'NormalRANSAC',
  normalNeighbors: 20,
  ransacThreshold: 0.01,
  maxIterations: 1000,
  normalDistanceWeight: 0.1,
  maxRadius: 11,
  minRadius: 6,
  axisOrientation: 'x',
  actualSpeed: 100,
  acquisitionSpeed: 100
};
// var 
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
      const settingsResponse = await axios.get('/api/settings/latest');
      const cylinderSettings = settingsResponse.data.data.cylinderSettings || {};
      const axis_now=cylinderSettings.axisOrientation|| 'x';
      setLoading(true);
      // console.log('settings.arcSettings',axis_now)
      message.info('axis_now:'+axis_now)
      //... 是展开设置对象，将对象的属性展开
      const response = await axios.post('/api/point-cloud/arc-fitting-stats', {
        ...settings.arcSettings,
        axis_now:axis_now
      },{
        timeout: 60000  // 60秒
      }
    );
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
          <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <div style={{ flex: 1 }}>
              <Statistic 
                title="总体中位数" 
                value={statistics.overallMedian} 
                precision={3}
                suffix="mm"
              />
            </div>
            <Tooltip title="复制数值">
              <Button
                type="default"
                icon={<CopyOutlined style={{ fontSize: '20px' }} />}
                onClick={() => {
                  const value = statistics.overallMedian.toFixed(3);
                  navigator.clipboard.writeText(value).then(() => {
                    message.success('已复制: ' + value);
                  });
                }}
                style={{ 
                  marginLeft: 12,
                  height: '38px',
                  width: '38px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: '8px',
                  borderRadius: '4px'
                }}
              />
            </Tooltip>
          </div>
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
      {/* 图表控制 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Button
            icon={<DownloadOutlined />}
            onClick={() => {
              // 创建CSV内容
              let csvContent = `总体中位数,时间\n${statistics.overallMedian},${new Date().toLocaleString()}\n\n`;
              csvContent += '图表数据\n序号,半径,类型,行号\n';
              
              // 添加当前图表数据
              chartData.forEach(point => {
                csvContent += `${point.index},${point.radius},${point.type},${point.lineIndex}\n`;
              });
              
              // 创建Blob并下载
              const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `圆弧拟合数据_${new Date().toLocaleString().replace(/[/:]/g, '-')}.csv`;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
              
              message.success('导出成功');
            }}
          >
            导出数据
          </Button>
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
