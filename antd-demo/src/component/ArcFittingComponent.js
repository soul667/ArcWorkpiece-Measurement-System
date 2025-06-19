import React, { useState, useEffect, useCallback } from 'react';
import { useCloud } from '../context/CloudContext';
import {
  Card,
  Table,
  Statistic,
  Row,
  Col,
  message,
  Radio,
  Space,
  Switch,
  Button,
  Tooltip,
  Select,
} from 'antd';
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
  MaxIterations: 1000,
  tolerance: 1e-6,
  fitIterations: 50, // 拟合迭代次数n
  samplePercentage: 50, // 采样百分比m
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
  acquisitionSpeed: 100,
};
// var
const ArcFittingComponent = () => {
  const { cloudName } = useCloud();
  const [lineData, setLineData] = useState([]);
  
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState(null);
  const [chartType, setChartType] = useState('line');
  const [showSorted, setshowSorted] = useState(false);
  const [statistics, setStatistics] = useState({
    overallMedian: 0,
    overallStd: 0,
    minValidRadius: 0,
    maxValidRadius: 0,
    totalValidCount: 0,
  });
  const [columnsPerRow, setColumnsPerRow] = useState(6);
  const [sortedRadii, setSortedRadii] = useState([]);

  // 将数据转换为多列格式
  const transformData = useCallback(
    (data) => {
      const transformedData = [];
      let currentRow = {};
      data.forEach((item, index) => {
        const columnIndex = index % columnsPerRow;
        currentRow[`lineIndex${columnIndex}`] = item.lineIndex;
        currentRow[`radius${columnIndex}`] = item.medianRadius;

        if (columnIndex === columnsPerRow - 1 || index === data.length - 1) {
          transformedData.push(currentRow);
          currentRow = {};
        }
      });
      return transformedData;
    },
    [columnsPerRow]
  );

  // 列定义
  // 列定义
  const columns = useCallback(() => {
    const cols = [];
    for (let i = 0; i < columnsPerRow; i++) {
      cols.push(
        {
          title: `线编号 ${i + 1}`,
          dataIndex: `lineIndex${i}`,
          key: `lineIndex${i}`,
          width: 80,
        },
        {
          title: `半径 ${i + 1}`,
          dataIndex: `radius${i}`,
          key: `radius${i}`,
          width: 100,
          render: (value) => (value ? value.toFixed(3) : ''),
        }
      );
    }
    return cols;
  }, [columnsPerRow]);

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
      const axis_now = cylinderSettings.axisOrientation || 'x';
      setLoading(true);
      console.log('settings.arcSettings', settingsResponse.data);
      message.info('axis_now:' + axis_now);
      //... 是展开设置对象，将对象的属性展开
      const response = await axios.post(
        '/api/point-cloud/arc-fitting-stats',
        {
          ...settings.arcSettings,
          axis_now: axis_now,
        },
        {
          timeout: 160000, // 60秒
        }
      );
      if (response.data.status === 'success') {
        setLineData(response.data.lineStats);
        setStatistics(response.data.overallStats);
        setSortedRadii(response.data.sortedRadii);
        // console.log('获取圆拟合数据成功:', response.data.sortedRadii);
        // console.log('排序后半径:', sortedRadii);
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
  const processChartData = useCallback(
    (data) => {
      let allRadiusData = [];
      data.forEach((line) => {
        if (line.radiusData && Array.isArray(line.radiusData)) {
          const linePoints = line.radiusData
            // .filter((point) => showSorted || point.isValid)
            .map((point) => ({
              ...point,
              lineIndex: line.lineIndex + 1,
              type: point.isValid ? '有效值' : '异常值',
              radius: point.radius,
              // radius: sortedRadii[point.iteration + 1]
            }));
          allRadiusData = [...allRadiusData, ...linePoints];
        }
      });

      // 根据图表类型处理数据
      switch (chartType) {
        case 'line':
            return allRadiusData.map((point, idx) => ({
            index: idx + 1,
            radius: showSorted
              ? (typeof sortedRadii[idx] !== 'undefined' ? sortedRadii[idx] : 0)
              : point.radius,
            type: point.type,
            lineIndex: point.lineIndex,
            iteration: point.iteration || idx,
            isValid: point.isValid,
            }));

        case 'stock':
          // 将数据分组，每10个点为一组
          const groupSize = 10;
          return Array.from({ length: Math.ceil(allRadiusData.length / groupSize) }, (_, i) => {
            const group = allRadiusData.slice(i * groupSize, (i + 1) * groupSize);
            const radii = group.map((p) => p.radius);
            return {
              index: i + 1,
              low: Math.min(...radii),
              high: Math.max(...radii),
              open: radii[0],
              close: radii[radii.length - 1],
              count: group.length,
            };
          });

        case 'scatter':
          return allRadiusData.map((point, idx) => ({
            index: idx + 1,
            radius: point.radius,
            type: point.type,
            lineIndex: point.lineIndex,
          }));

        case 'column':
          // 计算半径值的分布
          const binSize = 0.1; // 柱状图的区间大小
          const bins = {};
          allRadiusData.forEach((point) => {
            const bin = Math.floor(point.radius / binSize) * binSize;
            bins[bin] = (bins[bin] || 0) + 1;
          });
          return Object.entries(bins).map(([bin, count]) => ({
            radius: parseFloat(bin),
            count: count,
          }));

        default:
          return [];
      }
    },
    [chartType, showSorted]
  );

  // 获取当前图表配置
  const getChartConfig = () => {
    const baseConfig = {
      animation: {
        appear: {
          animation: 'wave-in',
          duration: 1000,
        },
      },
    };

    switch (chartType) {
      case 'line':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'index',
          yField: 'radius',
          // seriesField: 'type',
          color: ['#2196F3'],
          xAxis: { title: { text: '数据点序号' } },
          yAxis: { title: { text: '半径 (mm)' } },
          // legend: { position: 'top-right' },
          tooltip: {
            items: [
              (datum, index, data, column) => ({
                name: `${index}半径`,
                color: index === 100 ? 'red' : 'blue', // 指定 item 的颜色
                // name: index === 0 ? datum.genre : `${datum.genre} ${data[index].genre}`, // 指定 item 的名字
                value: `${data[index].radius.toFixed(3)}mm`, // 使用 y 通道的值
                // custom1: '自定义参数1',
                // custom2: '自定义参数2',
              }),
            ],
          },
          point: { size: 3, shape: 'circle' },
          // slider: {
          //   start: 0,
          //   end: 1,
          //   height: 20,
          // },
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
                `数量: ${datum.count}`,
            }),
          },
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
              name: '',
              value: `${datum.radius.toFixed(3)}`,
            }),
          },
          shape: 'circle',
          size: 4,
          slider: {
            start: 0,
            end: 1,
            height: 20,
          },
          meta: {
            radius: {
              nice: true,
            },
          },
          brush: {
            enabled: true,
          },
        };

      case 'column':
        return {
          ...baseConfig,
          data: chartData,
          xField: 'radius',
          yField: 'count',
          xAxis: {
            title: { text: '半径 (mm)' },
            label: { formatter: (v) => parseFloat(v).toFixed(2) },
          },
          yAxis: { title: { text: '数量' } },
          tooltip: {
            formatter: (datum) => ({
              name: '',
              value: `${datum.radius.toFixed(3)}`,
            }),
          },
          color: '#2196F3',
        };

      default:
        return baseConfig;
    }
  };
  const handleHistorySave = async () => {
    try {
      const record = {
        cloud_name: cloudName,
        timestamp: new Date().toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        }).replace(/\//g, '-'),
        radius: statistics.overallMedian,
        axis_vector_x: statistics.xMedian || 0,
        axis_vector_y: statistics.yMedian || 0,
        axis_vector_z: 0,
        axis_point_x: 0,
        axis_point_y: 0,
        axis_point_z: 0
      };

      await axios.post('/api/history/store', record);
      message.success('保存到历史记录成功');
    } catch (error) {
      console.error('保存到历史记录失败:', error);
      message.error('保存到历史记录失败');
    }
  }
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
        <Col span={4}>
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
                  height: '18px',
                  width: '18px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: '15px',
                  borderRadius: '10px',
                }}
              />
            </Tooltip>
          </div>
        </Col>
        <Col span={4}>
          <Statistic title="总体标准差" value={statistics.overallStd} precision={6} />
        </Col>
        <Col span={4}>
          <Statistic
            title="有效最小半径"
            value={statistics.minValidRadius}
            precision={3}
            suffix="mm"
          />
        </Col>
        <Col span={4}>
          <Statistic
            title="有效最大半径"
            value={statistics.maxValidRadius}
            precision={3}
            suffix="mm"
          />
        </Col>
        <Col span={4}>
          {/* <Statistic 
            title="圆心X" 
            value={statistics.xMedian} 
            precision={3}
            suffix="mm"
          /> */}
          <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <div style={{ flex: 1 }}>
              <Statistic title="圆心Y" value={statistics.xMedian} precision={3} suffix="mm" />
            </div>
            <Tooltip title="复制数值1">
              <Button
                type="default"
                icon={<CopyOutlined style={{ fontSize: '20px' }} />}
                onClick={() => {
                  const value = statistics.xMedian.toFixed(3);
                  navigator.clipboard.writeText(value).then(() => {
                    message.success('已复制: ' + value);
                  });
                }}
                style={{
                  height: '18px',
                  width: '18px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: '15px',
                  borderRadius: '10px',
                }}
              />
            </Tooltip>
          </div>
        </Col>
        <Col span={4}>
          {/* <Statistic 
            title="圆心Y" 
            value={statistics.yMedian} 
            precision={3}
            suffix="mm"
          /> */}
          <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <div style={{ flex: 1 }}>
              <Statistic title="圆心Y" value={statistics.yMedian} precision={3} suffix="mm" />
            </div>
            <Tooltip title="复制数值">
              <Button
                type="default"
                icon={<CopyOutlined style={{ fontSize: '20px' }} />}
                onClick={() => {
                  const value = statistics.yMedian.toFixed(3);
                  navigator.clipboard.writeText(value).then(() => {
                    message.success('已复制: ' + value);
                  });
                }}
                style={{
                  height: '18px',
                  width: '18px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: '15px',
                  borderRadius: '10px',
                }}
              />
            </Tooltip>
          </div>
        </Col>
      </Row>

      {/* 图表控制 */}
      {/* 图表控制 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Select
            value={columnsPerRow}
            onChange={setColumnsPerRow}
            options={[
              { value: 2, label: '每行2组' },
              { value: 3, label: '每行3组' },
              { value: 4, label: '每行4组' },
              { value: 5, label: '每行5组' },
              { value: 6, label: '每行6组' },
              { value: 7, label: '每行7组' },
              { value: 8, label: '每行8组' },
              { value: 9, label: '每行9组' },
              { value: 10, label: '每行10组' },
            ]}
            style={{ width: 120 }}
          />
          <Button
            icon={<DownloadOutlined />}
            onClick={() => {
              // 创建CSV内容
              let csvContent = `总体中位数,时间\n${
                statistics.overallMedian
              },${new Date().toLocaleString()}\n\n`;
              csvContent += '图表数据\n序号,半径,类型,行号\n';

              // 添加当前图表数据
              chartData.forEach((point) => {
                csvContent += `${point.index},${point.radius},${point.type},${point.lineIndex}\n`;
              });

              // 创建Blob并下载
              const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `圆弧拟合数据_${new Date()
                .toLocaleString()
                .replace(/[/:]/g, '-')}.csv`;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);

              message.success('导出成功');
            }}
          >
            导出数据
          </Button>
          <Radio.Group value={chartType} onChange={(e) => setChartType(e.target.value)}>
            <Radio.Button value="line">折线图</Radio.Button>
            {/* <Radio.Button value="stock">K线图</Radio.Button> */}
            <Radio.Button value="scatter">散点图</Radio.Button>
            <Radio.Button value="column">柱状图</Radio.Button>
          </Radio.Group>
          <Switch
            checkedChildren="排序后"
            unCheckedChildren="原始数据"
            checked={showSorted}
            onChange={setshowSorted}
          />
        </Space>

        <Button type="primary" onClick={handleHistorySave} style={{ marginLeft: 16 }}>
          保存到历史记录
        </Button>
      </div>

      {/* 图表展示 */}
      <div style={{ marginBottom: 24, height: 400 }}>{renderChart()}</div>

      {/* 数据表格 */}
      <Table
        columns={columns()}
        dataSource={transformData(lineData)}
        loading={loading}
        rowKey={(record, index) => index}
        pagination={{ pageSize: 10 }}
        scroll={{ x: 'max-content' }}
      />
    </Card>
  );
};

export default ArcFittingComponent;
