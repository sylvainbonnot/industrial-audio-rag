# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing [your@email.com]. 

**Please do not report security vulnerabilities through public GitHub issues.**

We will respond to security reports within 48 hours and provide regular updates on our progress.

## Security Considerations

This project processes audio data and integrates with external services. Key security considerations:

- **Data Privacy**: Audio files may contain sensitive information
- **API Security**: The FastAPI service should be deployed with proper authentication
- **Dependencies**: We regularly scan dependencies for vulnerabilities
- **Container Security**: Docker images use non-root users and minimal base images

## Responsible Disclosure

We appreciate security researchers who:
- Give us reasonable time to fix issues before public disclosure
- Avoid accessing or modifying data that doesn't belong to you
- Don't perform actions that could harm the service or other users

Thank you for helping keep our project secure!