# Security Policy

## Reporting

If GitHub private vulnerability reporting is enabled for the repository, use it for security-sensitive issues. Otherwise, open a minimal public issue that does not disclose exploit details and ask the maintainer to establish a private channel.

Report concerns involving:

- Dependency vulnerabilities
- FastAPI inference-service security issues
- Secret exposure
- Unsafe model behavior or misleading maintenance recommendations
- Path traversal or file-handling issues

## Secrets

Do not commit `.env` files, credentials, private keys, API tokens or cloud credentials. Use `.env.example` for placeholders only.

## Model Safety

AeroGuard-PHM is a research and portfolio system. It is not certified for aviation maintenance, dispatch or safety-critical operational control. Unsafe behavior should be reported as a model-governance issue even when it is not a software exploit.

