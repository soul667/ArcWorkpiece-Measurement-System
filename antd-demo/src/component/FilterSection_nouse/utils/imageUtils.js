import axios from '../../../utils/axios';

/**
 * Load and process image from server
 * @param {string} view View type ('xy', 'xz', 'yz')
 * @returns {Promise<string>} Image URL
 */
export const loadImage = async (view) => {
  try {
    const response = await axios.get(`/api/files/img/${view}?t=${Date.now()}`, {
      responseType: 'blob'
    });
    const blob = new Blob([response.data], { type: 'image/jpeg' });
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error('加载图像失败:', error);
    throw new Error('加载图像失败，请检查认证状态');
  }
};

/**
 * Parse YAML coordinate information
 * @param {string} yamlText Raw YAML text
 * @returns {Object} Parsed coordinate ranges
 */
export const parseYaml = (yamlText) => {
  const lines = yamlText.split('\n').filter((line) => line.includes(':'));
  const result = {};
  lines.forEach((line) => {
    const [key, value] = line.split(':').map((str) => str.trim());
    result[key] = parseFloat(value);
  });
  return result;
};

/**
 * Fetch coordinate ranges from server
 * @returns {Promise<Object>} Coordinate ranges
 */
export const fetchCoordinateRanges = async () => {
  try {
    const response = await axios.get(`/api/files/yml/info?t=${Date.now()}`);
    return parseYaml(response.data);
  } catch (error) {
    console.error('Error fetching YAML:', error);
    throw error;
  }
};

/**
 * Normalize regions based on coordinate ranges
 * @param {Array} regions Array of region pairs
 * @param {string} axis Axis ('x', 'y', or 'z')
 * @param {Object} coordinateRanges Range information
 * @returns {Array} Normalized regions
 */
export const normalizeRegions = (regions, axis, coordinateRanges) => {
  const min = coordinateRanges[`${axis}_min`];
  const max = coordinateRanges[`${axis}_max`];
  
  return regions.map(([start, end]) => [
    min + (max - min) * start,
    min + (max - min) * end
  ]);
};

/**
 * Transform regions for different views
 * @param {Object} params Region and view information
 * @returns {Object} Transformed regions
 */
export const transformRegions = ({ 
  xRegions, 
  yRegions, 
  currentView, 
  coordinateRanges 
}) => {
  let regions;
  switch (currentView) {
    case 'xy':
      regions = {
        x_regions: normalizeRegions(xRegions, 'x', coordinateRanges),
        y_regions: normalizeRegions(yRegions, 'y', coordinateRanges),
        z_regions: []
      };
      break;
    case 'xz':
      regions = {
        x_regions: normalizeRegions(xRegions, 'x', coordinateRanges),
        y_regions: [],
        z_regions: normalizeRegions(yRegions, 'z', coordinateRanges)
      };
      break;
    case 'yz':
      regions = {
        x_regions: [],
        y_regions: normalizeRegions(xRegions, 'y', coordinateRanges),
        z_regions: normalizeRegions(yRegions, 'z', coordinateRanges)
      };
      break;
    default:
      regions = {
        x_regions: [],
        y_regions: [],
        z_regions: []
      };
  }
  return regions;
};
