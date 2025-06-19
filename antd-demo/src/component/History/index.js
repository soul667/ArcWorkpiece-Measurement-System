import React, { useState, useEffect } from 'react';
import { 
  Table, Card, Space, Button, Row, Col, 
  Image, Statistic, Typography, Drawer,
  Descriptions, Divider, Input, message
} from 'antd';
import { FileOutlined } from '@ant-design/icons';
import axios from '../../utils/axios';

const { Text } = Typography;

const HistorySection = () => {
  const [records, setRecords] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [editingRecord, setEditingRecord] = useState(null);
  const [detailVisible, setDetailVisible] = useState(false);
  const [statistics, setStatistics] = useState(null);

  const handleSaveName = async (record, newName) => {
    if (newName !== record.cloud_name) {
      try {
        await axios.put(`/api/history/${record.timestamp}`, {
          cloud_name: newName
        });
        message.success('名称更新成功');
        fetchRecords();
      } catch (error) {
        message.error('名称更新失败');
      }
    }
    setEditingRecord(null);
  };

  // 获取历史记录
  useEffect(() => {
    fetchRecords();
  }, []);

  const fetchRecords = async () => {
    try {
      const response = await axios.get('/api/history/list');
      if (response.data.status === 'success') {
        setRecords(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch records:', error);
    }
  };

  // 计算选中记录的统计数据
  useEffect(() => {
    if (selectedRows.length > 0) {
      const radiusSum = selectedRows.reduce((sum, row) => sum + row.radius, 0);
      const averageRadius = radiusSum / selectedRows.length;
      
      // 计算标准差
      const squaredDiffs = selectedRows.map(row => 
        Math.pow(row.radius - averageRadius, 2)
      );
      const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / selectedRows.length;
      const stdDev = Math.sqrt(variance);

      // 计算轴线方向均值
      const axisVectorSum = selectedRows.reduce((sum, row) => ({
        x: sum.x + row.axis_vector_x,
        y: sum.y + row.axis_vector_y,
        z: sum.z + row.axis_vector_z
      }), { x: 0, y: 0, z: 0 });

      setStatistics({
        averageRadius,
        stdDev,
        axisVector: {
          x: axisVectorSum.x / selectedRows.length,
          y: axisVectorSum.y / selectedRows.length,
          z: axisVectorSum.z / selectedRows.length
        }
      });
    } else {
      setStatistics(null);
    }
  }, [selectedRows]);

  // 处理批量删除
  const handleBatchDelete = async () => {
    if (!selectedRows.length) return;
    
    try {
      for (const record of selectedRows) {
        await axios.delete(`/api/history/${record.timestamp}`);
      }
      fetchRecords();
      setSelectedRows([]);
    } catch (error) {
      console.error('Failed to delete records:', error);
    }
  };

  // 导出报告
  const handleExportReport = async () => {
    if (!selectedRows.length) {
      message.warning('请选择要导出的记录');
      return;
    }

    try {
      const response = await axios.post('/api/history/export', 
        { record_ids: selectedRows.map(row => row.id) },
        { 
          responseType: 'blob',
          headers: {
            'Accept': 'application/pdf'
          }
        }
      );

      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `测量报告_${new Date().toLocaleString().replace(/[/:]/g, '-')}.pdf`;
      document.body.appendChild(link);
      link.click();
      URL.revokeObjectURL(url);
      document.body.removeChild(link);
      message.success('导出成功');
    } catch (error) {
      if (error.response?.status === 401) {
        message.error('请先登录');
      } else {
        message.error('导出失败: ' + (error.response?.data?.detail || error.message));
        console.error('导出失败:', error);
      }
    }
  };

  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      sorter: true
    },
    {
      title: '测量名称',
      dataIndex: 'cloud_name',
      render: (text, record) => (
        <Space>
          {editingRecord?.timestamp === record.timestamp ? (
            <Input
              value={editingRecord.newName}
              onChange={e => setEditingRecord({
                ...editingRecord,
                newName: e.target.value || ''
              })}
              onPressEnter={() => handleSaveName(record, editingRecord.newName)}
              onBlur={() => handleSaveName(record, editingRecord.newName)}
              onKeyDown={e => {
                if (e.key === 'Escape') {
                  setEditingRecord(null);
                }
              }}
              autoFocus
            />
          ) : (
            <>
              <Typography.Text>{text}</Typography.Text>
              <Button
                type="link"
                size="small"
                onClick={() => setEditingRecord({
                  timestamp: record.timestamp,
                  newName: text || ''
                })}
              >
                修改名称
              </Button>
            </>
          )}
        </Space>
      )
    },
    {
      title: '拟合半径',
      dataIndex: 'radius',
      render: (radius) => `${radius.toFixed(3)} mm`
    },
    {
      title: '操作',
      render: (record) => (
        <Space size="middle">
          <Button type="link" onClick={() => {
            setSelectedRecord(record);
            setDetailVisible(true);
          }}>
            查看
          </Button>
          <Button 
            type="link" 
            danger 
            onClick={async () => {
              try {
                await axios.delete(`/api/history/${record.timestamp}`);
                fetchRecords();
              } catch (error) {
                console.error('Failed to delete record:', error);
              }
            }}
          >
            删除
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 工具栏 */}
      <Card style={{ marginBottom: 16 }}>
        <Space>
          <Button 
            type="primary" 
            disabled={selectedRows.length < 2}
            onClick={() => {
              if (statistics) {
                setSelectedRecord({
                  cloud_name: '均值计算结果',
                  timestamp: new Date().toISOString(),
                  radius: statistics.averageRadius,
                  axis_vector_x: statistics.axisVector.x,
                  axis_vector_y: statistics.axisVector.y,
                  axis_vector_z: statistics.axisVector.z,
                });
                setDetailVisible(true);
              }
            }}
          >
            计算均值
          </Button>
          <Button 
            danger 
            disabled={!selectedRows.length}
            onClick={handleBatchDelete}
          >
            批量删除
          </Button>
          {/* <Button 
            type="primary"
            disabled={!selectedRows.length}
            icon={<FileOutlined />}
            onClick={handleExportReport}
          >
            导出报告
          </Button> */}
        </Space>
      </Card>

      <Row gutter={16}>
        {/* 主数据表格 */}
        <Col span={18}>
          <Table
            rowSelection={{
              type: 'checkbox',
              onChange: (_, rows) => setSelectedRows(rows)
            }}
            columns={columns}
            dataSource={records}
            rowKey="id"
            pagination={{ pageSize: 10 }}
          />
        </Col>

        {/* 统计面板 */}
        <Col span={6}>
          {selectedRows.length > 0 && statistics && (
            <Card title="统计信息">
              <Statistic
                title="选中记录数"
                value={selectedRows.length}
              />
              <Statistic
                title="半径均值"
                value={statistics.averageRadius}
                precision={3}
                suffix="mm"
              />
              <Statistic
                title="标准差"
                value={statistics.stdDev}
                precision={3}
                suffix="mm"
              />
              <Divider />
              <Typography.Title level={5}>轴线方向均值</Typography.Title>
              <Space direction="vertical">
                <Text>X: {statistics.axisVector.x.toFixed(3)}</Text>
                <Text>Y: {statistics.axisVector.y.toFixed(3)}</Text>
                <Text>Z: {statistics.axisVector.z.toFixed(3)}</Text>
              </Space>
            </Card>
          )}
        </Col>
      </Row>

      {/* 详情抽屉 */}
      <Drawer
        title="测量记录详情"
        width={600}
        open={detailVisible}
        onClose={() => {
          setDetailVisible(false);
          setEditingRecord(null);
        }}
      >
        {selectedRecord && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="测量名称">
                {selectedRecord.cloud_name}
              </Descriptions.Item>
              <Descriptions.Item label="时间戳">
                {selectedRecord.timestamp}
              </Descriptions.Item>
              <Descriptions.Item label="拟合半径">
                {selectedRecord.radius.toFixed(3)} mm
              </Descriptions.Item>
            </Descriptions>

            <Card title="轴线信息">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Statistic 
                    title="方向向量" 
                    value={`(${selectedRecord.axis_vector_x.toFixed(3)}, 
                           ${selectedRecord.axis_vector_y.toFixed(3)}, 
                           ${selectedRecord.axis_vector_z.toFixed(3)})`}
                  />
                </Col>
                <Col span={12}>
                  <Statistic 
                    title="轴线点" 
                    value={`(${selectedRecord.axis_point_x.toFixed(3)}, 
                           ${selectedRecord.axis_point_y.toFixed(3)}, 
                           ${selectedRecord.axis_point_z.toFixed(3)})`}
                  />
                </Col>
              </Row>
            </Card>

          </Space>
        )}
      </Drawer>
    </div>
  );
};

export default HistorySection;
