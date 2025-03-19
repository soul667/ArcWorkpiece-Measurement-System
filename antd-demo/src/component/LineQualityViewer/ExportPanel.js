import React from 'react';
import { Button, message, Space, Tooltip, Modal } from 'antd';
import { DownloadOutlined, DeleteOutlined } from '@ant-design/icons';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import axios from 'axios';

const ExportPanel = ({ data, defectLines, loading, total }) => {
  const handleExport = () => {
    try {
      if (!data?.group?.points) {
        message.error('没有可导出的数据');
        return;
      }

      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:-]/g, '')
        .replace('T', '_')
        .split('.')[0];

      // 准备导出数据
      const exportData = {
        id: data.current_index,
        timestamp: timestamp,
        // arc_points: data.group.points.map(point => point[2]),
        x: data.group.points.map(point => point[0]),
        y: data.group.points.map(point => point[1]),
        z: data.group.points.map(point => point[2]),
        label: defectLines[data.current_index] ? 1 : 0
      };

      // 创建并下载文件
      const blob = new Blob(
        [JSON.stringify(exportData, null, 2)], 
        { type: 'application/json' }
      );
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `line_${data.current_index}_${timestamp}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      message.success('数据导出成功');
    } catch (error) {
      console.error('导出失败:', error);
      message.error('数据导出失败');
    }
  };

  const handleExportAll = async () => {
    try {
      const zip = new JSZip();
      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:-]/g, '')
        .replace('T', '_')
        .split('.')[0];
        // /api/point-cloud/denoise
      // 获取所有线条数据
      const promises = Array.from({ length: total }, async (_, index) => {
        try {
          const response = await axios.post('/api/point-cloud/group-points', { 
            axis: data.axis,
            index
          });
          // console.log
          const lineData = {
            id: index,
            timestamp: timestamp,
            x: response.data.group.points.map(point => point[0]),
            y: response.data.group.points.map(point => point[1]),
            z: response.data.group.points.map(point => point[2]),
            label: defectLines[index] ? 1 : 0
          };
          
          zip.file(`line_${index}_${timestamp}.json`, JSON.stringify(lineData, null, 2));
        } catch (error) {
          console.error(`获取线条 ${index} 数据失败:`, error);
          throw error;
        }
      });

      await Promise.all(promises);

      const content = await zip.generateAsync({ type: "blob" });
      saveAs(content, `all_lines_${timestamp}.zip`);

      message.success('所有数据导出成功');
    } catch (error) {
      console.error('导出所有数据失败:', error);
      message.error('导出所有数据失败');
    }
  };

  const handleRemoveDefectLines = async () => {
    // 收集所有标记为缺陷的线条索引
    const defectIndices = defectLines.reduce((acc, isDefect, index) => {
      if (isDefect) acc.push(index);
      return acc;
    }, []);

    if (defectIndices.length === 0) {
      message.info('没有标记的缺陷线条');
      return;
    }

    Modal.confirm({
      title: '确认删除',
      content: `确定要删除${defectIndices.length}条标记为缺陷的线条吗？此操作不可恢复。`,
      okText: '确认删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const response = await axios.post('/api/remove-defect-lines', {
            defect_indices: defectIndices
          });
          message.success('缺陷线条已删除，正在重新生成预处理文件');
          // 刷新页面以加载新数据
          window.location.reload();
        } catch (error) {
          console.error('删除缺陷线条失败:', error);
          message.error('删除缺陷线条失败');
        }
      }
    });
  };

  return (
    <Space>
      <Button 
        type="primary" 
        icon={<DownloadOutlined />} 
        onClick={handleExport}
        disabled={!data?.group?.points || loading}
      >
        导出当前线条
      </Button>
      <Button
        type="primary"
        icon={<DownloadOutlined />}
        onClick={handleExportAll}
        disabled={loading}
      >
        导出全部
      </Button>
      <Button
        type="primary"
        danger
        icon={<DeleteOutlined />}
        onClick={handleRemoveDefectLines}
        disabled={loading}
      >
        删除缺陷线条
      </Button>
    </Space>
  );
};

export default ExportPanel;
