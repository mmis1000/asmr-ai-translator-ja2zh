/** Sanitize path-unsafe characters to percent encoding. */
export const sanitizePath = (path: string) => {
  return path.replace(/[%<>:"/\\|?*\x00-\x1F]/g, (char) => {
    return "%" + char.charCodeAt(0).toString(16).toUpperCase().padStart(2, "0");
  });
};
