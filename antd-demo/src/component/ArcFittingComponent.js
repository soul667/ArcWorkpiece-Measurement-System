import React, { useState, useEffect } from 'react';
import { Card, Table, Statistic, Row, Col, message } from 'antd';
import axios from '../utils/axios';

const ArcFittingComponent = () => {
  const [lineData, setLineData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState({
    overallMean: 0,
    overallStd: 0,
    minRadius: 0,
    maxRadius: 0
  });

  // 列定义
  const columns = [
    {
      title: '线编号',
      dataIndex: 'lineIndex',
      key: 'lineIndex',
    },
    {
      title: '迭代半径',
      dataIndex: 'radii',
      key: 'radii',
      render: (radii) => radii.map(r => r.toFixed(3)).join(', ')
    },
    {
      title: '平均半径',
      dataIndex: 'meanRadius',
      key: 'meanRadius',
      render: (mean) => mean.toFixed(3)
    },
    {
      title: '标准差',
      dataIndex: 'stdDev',
      key: 'stdDev',
      render: (std) => std.toFixed(6)
    }
  ];

  // 从API获取数据
  const fetchData = async () => {
    try {
      setLoading(true);
      const settings = {
        arcNormalNeighbors: 10,
        fitIterations: 50,
        samplePercentage: 50
      };
      
      const response = await axios.post('/api/arc-fitting-stats', settings);
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
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <Card title="圆弧拟合结果" bordered={false}>
      {/* 统计信息展示 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Statistic 
            title="总体平均半径" 
            value={statistics.overallMean} 
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
            title="最小半径" 
            value={statistics.minRadius} 
            precision={3}
            suffix="mm"
          />
        </Col>
        <Col span={6}>
          <Statistic 
            title="最大半径" 
            value={statistics.maxRadius} 
            precision={3}
            suffix="mm"
          />
        </Col>
      </Row>

      {/* 数据表格 */}
      <Table
        columns={columns}
        dataSource={lineData}
        loading={loading}
        rowKey="lineIndex"
        scroll={{ y: 400 }}
      />
    </Card>
  );
};

export default ArcFittingComponent;
