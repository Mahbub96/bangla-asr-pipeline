export function AudioPreview({ url, mimeType }: { url: string; mimeType: string }) {
  if (!url) return null;
  if (mimeType.startsWith("video/")) return <video className="audio-preview media-preview" src={url} controls />;
  return <audio className="audio-preview media-preview" src={url} controls />;
}
