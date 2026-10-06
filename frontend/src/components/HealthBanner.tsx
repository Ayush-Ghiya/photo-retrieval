import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function HealthBanner() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000, retry: false });
  if (!health.isError) return null;
  return (
    <div role="alert" className="bg-red-600 px-4 py-2 text-center text-sm text-white">
      Backend services unavailable — is <code>docker compose up -d</code> running and the API started?
    </div>
  );
}
