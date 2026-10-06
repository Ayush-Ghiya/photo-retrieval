export type Source = "upload" | "demo";

export interface ImageSummary {
  id: string;
  filename: string;
  title: string | null;
  description: string | null;
  tags: string[];
  width: number;
  height: number;
  taken_at: string | null;
  created_at: string;
  source: Source;
  indexed: boolean;
  thumb_url: string;
}

export interface ImageDetail extends ImageSummary {
  original_url: string;
  mime_type: string;
}

export interface ImagePage {
  items: ImageSummary[];
  page: number;
  page_size: number;
  total: number;
}

export interface SearchItem extends ImageSummary {
  score: number | null;
}

export interface SearchResponse {
  items: SearchItem[];
}

export interface TagCount {
  name: string;
  count: number;
}

export interface UploadResult {
  filename: string;
  status: "created" | "duplicate" | "error";
  id: string | null;
  message: string | null;
}

export interface ImagePatch {
  title?: string | null;
  description?: string | null;
  tags?: string[];
}

export interface UpdateResponse {
  image: ImageDetail;
  warning: string | null;
}

export interface UploadMeta {
  tags: string[];
  title: string;
  description: string;
}

export interface ListParams {
  page: number;
  page_size: number;
  tags?: string[];
  source?: Source;
  sort?: "taken" | "uploaded";
}

export interface SearchParams {
  q: string;
  tags?: string[];
  source?: Source;
  limit?: number;
}
