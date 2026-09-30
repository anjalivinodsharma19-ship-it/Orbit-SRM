export function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading">
    <div>
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
    {action}
  </div>;
}

export function Status({ value }) {
  const status = value || 'unknown';
  return <span className={`status status-${status}`} role="status">
    <i aria-hidden="true" />{status}
  </span>;
}

export function StatCard({ icon: Icon, tone, label, value, detail }) {
  return <article className="stat-card">
    <div className={`stat-icon ${tone}`} aria-hidden="true"><Icon size={18} /></div>
    <div className="stat-label">{label}</div>
    <div className="stat-value">{value}</div>
    <div className="stat-detail">{detail}</div>
  </article>;
}

export function InfoRow({ label, value, state }) {
  return <div className="info-row">
    <span>{label}</span>
    <b className={state === false ? 'muted-value' : state ? 'good-value' : ''}>{value}</b>
  </div>;
}