/**
 * Canvas utility functions for drawing and calculations
 */

/**
 * Calculate canvas dimensions based on container and image
 * @param {HTMLElement} container Canvas container element
 * @param {Image} image Image element
 * @param {boolean} isFullscreen Whether canvas is in fullscreen mode
 * @returns {Object} Canvas dimensions and scale
 */
export const calculateCanvasDimensions = (container, image, isFullscreen) => {
  const containerWidth = isFullscreen ? window.innerWidth * 0.8 : container.clientWidth;
  const containerHeight = isFullscreen ? window.innerHeight * 0.8 : window.innerHeight * 0.7;
  
  const imageAspectRatio = image.width / image.height;
  const containerAspectRatio = containerWidth / containerHeight;
  
  let width, height;
  if (imageAspectRatio > containerAspectRatio) {
    width = containerWidth;
    height = containerWidth / imageAspectRatio;
  } else {
    height = containerHeight;
    width = containerHeight * imageAspectRatio;
  }
  
  return {
    width,
    height,
    scale: window.devicePixelRatio
  };
};

/**
 * Draw image and regions on canvas
 * @param {CanvasRenderingContext2D} ctx Canvas context
 * @param {Image} image Image to draw
 * @param {Array} xRegions Vertical regions
 * @param {Array} yRegions Horizontal regions
 * @param {Array} lines Current drawing lines
 * @param {string} drawMode Current drawing mode ('x' or 'y')
 * @param {number} width Canvas width
 * @param {number} height Canvas height
 */
export const drawCanvas = (ctx, image, xRegions, yRegions, lines, drawMode, width, height) => {
  // Clear canvas
  ctx.clearRect(0, 0, width, height);
  
  // Draw image
  ctx.drawImage(image, 0, 0, width, height);

  // Draw vertical regions
  xRegions.forEach((region) => {
    const [x1, x2] = region;
    const pixelX1 = x1 * width;
    const pixelX2 = x2 * width;
    
    ctx.fillStyle = 'rgba(0, 0, 255, 0.1)';
    ctx.fillRect(pixelX1, 0, pixelX2 - pixelX1, height);
    
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pixelX1, 0);
    ctx.lineTo(pixelX1, height);
    ctx.moveTo(pixelX2, 0);
    ctx.lineTo(pixelX2, height);
    ctx.stroke();
  });

  // Draw horizontal regions
  yRegions.forEach((region) => {
    const [y1, y2] = region;
    const pixelY1 = y1 * height;
    const pixelY2 = y2 * height;
    
    ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
    ctx.fillRect(0, pixelY1, width, pixelY2 - pixelY1);
    
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, pixelY1);
    ctx.lineTo(width, pixelY1);
    ctx.moveTo(0, pixelY2);
    ctx.lineTo(width, pixelY2);
    ctx.stroke();
  });

  // Draw current lines
  lines.forEach(line => {
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (drawMode === 'x') {
      const pixelX = line.x * width;
      ctx.moveTo(pixelX, 0);
      ctx.lineTo(pixelX, height);
    } else {
      const pixelY = line.y * height;
      ctx.moveTo(0, pixelY);
      ctx.lineTo(width, pixelY);
    }
    ctx.stroke();
  });
};

/**
 * Convert mouse coordinates to relative canvas position
 * @param {MouseEvent} event Mouse event
 * @param {HTMLCanvasElement} canvas Canvas element
 * @returns {Object} Relative x and y coordinates
 */
export const getRelativeCoordinates = (event, canvas) => {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width,
    y: (event.clientY - rect.top) / rect.height
  };
};
