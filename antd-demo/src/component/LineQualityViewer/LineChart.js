import React, { useMemo } from 'react';
import { Scatter, Line } from '@ant-design/charts';
import { Empty, Spin } from 'antd';

const LineChart = ({ 
  data, 
  loading
}) => {
  const pointSize = 0.8
  const transformedData = useMemo(() => {
    if (!data?.group?.points) return [];
    
    const points = data.group.points;
    // 将3D点转换为2D视图
    return points.map((point, pointIndex) => ({
      lineIndex: 0,
      coordinate: data.group.coordinate,
      pointIndex: pointIndex,
      y: point[1], // y坐标
      z: point[2], // z坐标
    }));
  }, [data]);

  if (loading) {
    return (
      <div style={{ 
        height: '400px', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center' 
      }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!data?.group?.points || data.group.points.length === 0) {
    return <Empty description="暂无数据" />;
  }

  // 计算坐标范围，确保Y和Z轴使用相同的比例尺
  const yValues = transformedData.map(d => d.y);
  const zValues = transformedData.map(d => d.z);
  
  // 分别计算Y轴和Z轴的数据范围
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const zMin = Math.min(...zValues);
  const zMax = Math.max(...zValues);
  
  // 计算数据范围
  const yRange = yMax - yMin;
  const zRange = zMax - zMin;
  const ratio= yRange / zRange;
  // 确定较大的范围并添加统一的边距
  const maxRange = Math.max(yRange, zRange);
  const padding = maxRange * 0.1;
  const totalRange = maxRange + padding * 2;
  
  // 计算各轴的中点
  const yMid = (yMin + yMax) / 2;
  const zMid = (zMin + zMax) / 2;
  
  // 基于中点和统一范围计算各轴的显示范围
  const yDomain = [yMid - totalRange/2, yMid + totalRange/2];
  const zDomain = [zMid - totalRange/2, zMid + totalRange/2];

  const config = {
    data: transformedData,
    sizeField: pointSize,
    xField: 'y',
    yField: 'z',
    seriesField: 'lineIndex',
    color: '#d9d9d9',
    shape: 'circle',
    pointStyle: {
      fill: '#d9d9d9',
      r: pointSize,
      opacity: 0.7,
      stroke: '#bfbfbf',
      lineWidth: 1
    },
    tooltip: {
      showTitle: true,
      title: `第${data?.current_index + 1}条线`,
      showMarkers: false,
      fields: ['coordinate', 'y', 'z'],
      formatter: (datum) => {
        return [
          { name: '位置', value: datum.coordinate.toFixed(4) },
          { name: 'Y', value: datum.y.toFixed(4) },
          { name: 'Z', value: datum.z.toFixed(4) }
        ];
      }
    },
    appendPadding: [35, 35, 35, 35],
    width: 1000,
    height: 200,
    meta: {
      y: { 
        min: yDomain[0], 
        max: yDomain[1],
        // nice: false  // 禁用自动调整
      },
      z: { 
        min: zDomain[0], 
        max: zDomain[1]
      },
      coordinate: { formatter: (v) => v.toFixed(3) }
    },
    xAxis: {
      min: yDomain[0],
      max: yDomain[1]
    },
    yAxis: {
      min: zDomain[0],
      max: zDomain[1]
    },
    axes: {
      y: {
        title: 'Y轴',
        grid: { 
          line: { 
            style: { 
              stroke: '#f0f0f0', 
              lineDash: [4, 4] 
            } 
          } 
        },
        line: { style: { stroke: '#d9d9d9' } },
        label: { 
          formatter: (v) => v.toFixed(2),
          style: {
            fill: '#666'
          }
        }
      },
      z: {
        title: 'Z轴',
        grid: { 
          line: { 
            style: { 
              stroke: '#f0f0f0', 
              lineDash: [4, 4] 
            } 
          } 
        },
        line: { style: { stroke: '#d9d9d9' } },
        label: { 
          formatter: (v) => v.toFixed(2),
          style: {
            fill: '#666'
          }
        }
      }
    },
    interactions: [
      { type: 'element-active' },
      { type: 'brush' }
    ],
    animation: false,
    theme: {
      components: {
        axis: {
          title: {
            autoRotate: true,
            style: {
              fill: '#666',
              fontSize: 12
            }
          }
        }
      }
    },
    aspectRatio: ratio, 
  };

  return (
      <Scatter {...config} />
  );
};

export default LineChart;
