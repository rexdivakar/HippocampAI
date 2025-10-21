# Security Policy

## Supported Versions

We currently support the following versions of HippocampAI with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities to:

- **Email**: rexdivakar@hotmail.com
- **Subject**: [SECURITY] HippocampAI - [Brief Description]

Please include the following information in your report:

1. **Type of vulnerability** (e.g., injection, authentication bypass, data exposure)
2. **Affected component(s)** (e.g., API endpoint, configuration, dependencies)
3. **Steps to reproduce** the vulnerability
4. **Potential impact** of the vulnerability
5. **Suggested fix** (if you have one)

### What to Expect

- **Initial Response**: You should receive an acknowledgment within 48 hours
- **Status Updates**: We will keep you informed about the progress every 5-7 days
- **Resolution Timeline**: We aim to fix critical vulnerabilities within 7 days
- **Disclosure**: We will coordinate with you on responsible disclosure timing

### Security Best Practices

When using HippocampAI in production, please follow these security guidelines:

#### 1. **Environment Variables**
- Never commit `.env` files to version control
- Use secure secret management (AWS Secrets Manager, HashiCorp Vault, etc.)
- Rotate credentials regularly

#### 2. **Network Security**
- Use TLS/SSL for all network communication
- Restrict Qdrant access to trusted networks only
- Enable authentication on Qdrant in production
- Use firewall rules to limit access

#### 3. **Input Validation**
- Validate and sanitize all user inputs
- Set reasonable limits on memory text length
- Validate user_id and session_id formats
- Implement rate limiting per user

#### 4. **Data Protection**
- Encrypt sensitive data at rest
- Use encrypted connections to Qdrant
- Implement proper access controls
- Regular backups with encryption

#### 5. **Dependencies**
- Regularly update dependencies to patch vulnerabilities
- Run `pip-audit` or `safety check` in CI/CD
- Review dependency licenses and security advisories

#### 6. **Authentication & Authorization**
- Implement authentication if exposing as a service
- Enforce user isolation (multi-tenancy)
- Use API keys or OAuth 2.0 for API access
- Implement role-based access control (RBAC)

#### 7. **Logging & Monitoring**
- Enable audit logging for all operations
- Monitor for suspicious activity patterns
- Set up alerts for anomalies
- Regularly review logs

#### 8. **Configuration**
- Disable `allow_cloud` if not needed
- Review and harden default configurations
- Use environment-specific configs (dev/staging/prod)
- Limit exposed API endpoints

### Known Security Considerations

1. **No Built-in Authentication**: HippocampAI does not include authentication by default. You must implement authentication if exposing it as a service.

2. **User Data Isolation**: While user data is separated by `user_id`, there's no built-in authorization layer. Implement proper access controls in your application.

3. **LLM API Keys**: If using cloud LLM providers (OpenAI, Anthropic), ensure API keys are stored securely.

4. **Graph Import**: The `import_graph_from_json` function deserializes JSON data. Only import graphs from trusted sources.

5. **Memory Size Limits**: No hard limits on memory text size by default. Set appropriate limits to prevent resource exhaustion.

### Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1 → 0.1.2)
- Documented in the [CHANGELOG](docs/CHANGELOG.md)
- Announced on GitHub releases
- Communicated via security advisories

### Vulnerability Disclosure Policy

We follow responsible disclosure practices:
1. Security vulnerabilities will be fixed before public disclosure
2. We will credit security researchers (unless they wish to remain anonymous)
3. We will coordinate disclosure timing with the reporter
4. CVE IDs will be requested for significant vulnerabilities

### Security Checklist for Production

Before deploying to production, ensure:

- [ ] All dependencies are up to date
- [ ] Security scanning is enabled in CI/CD
- [ ] Environment variables are properly secured
- [ ] Qdrant has authentication enabled
- [ ] Network access is restricted
- [ ] Rate limiting is implemented
- [ ] Audit logging is enabled
- [ ] Backups are encrypted and tested
- [ ] Monitoring and alerting are configured
- [ ] Incident response plan is in place

### Contact

For security-related questions or concerns:
- **Security Email**: rexdivakar@hotmail.com
- **GitHub**: https://github.com/rexdivakar/HippocampAI/security

Thank you for helping keep HippocampAI and our users safe!
