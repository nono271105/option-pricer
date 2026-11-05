const DAY_MS = 24 * 60 * 60 * 1000;

export function computeDaysFromDate(dateStr: string): number {
  const target = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(target.getTime())) return 1;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.max(Math.round((target.getTime() - today.getTime()) / DAY_MS), 1);
}

export function formatDateFR(dateStr: string): string {
  const parts = dateStr.split('-');
  if (parts.length !== 3) return dateStr;
  return `${parts[2]}/${parts[1]}/${parts[0]}`;
}

export function getDefaultMaturity(days = 90): string {
  const date = new Date();
  date.setDate(date.getDate() + days);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}
